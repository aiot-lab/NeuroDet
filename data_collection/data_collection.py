#############################################
# Author: Luca Yu
# Date: 2024-08-12
# Description: This script is used to collect data from the mmWave radar and depth camera, 
#              with optional visualization and data saving.
#############################################

import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from loguru import logger
import time
from mmwave.dataloader import DCA1000
from mmwave.dataloader.radars import TI
import datetime
import io
from radar_processing import processing
import pickle
import os
import argparse
import keyboard

# Set up input parameters
parser = argparse.ArgumentParser(description='Process radar and camera data.')
parser.add_argument('--user', type=str, required=True, help='Name of the volunteer')
parser.add_argument('--location', type=str, required=True, help='Location of the test')
parser.add_argument('--visualization', type=bool, default=False, help='Enable visualization')
parser.add_argument('--save_data', type=bool, default=False, help='Enable saving data')
parser.add_argument('--cli_loc', type=str, default='COM5', help='CLI port location')
parser.add_argument('--data_loc', type=str, default='COM6', help='Data port location')
args = parser.parse_args()

volunteer_name = args.user
test_location = args.location
enable_visualization = args.visualization
save_data = args.save_data
cli_loc = args.cli_loc
data_loc = args.data_loc
isReverse = True    # Set to True if the depth camera is vetically flipped

# Initialize global stop flag
stop_flag = False

def on_esc_key_press():
    """Handle the Esc key press to stop data collection."""
    global stop_flag
    stop_flag = True
    logger.info("Esc key pressed, stopping data collection...")

# Register Esc key listener
keyboard.add_hotkey('esc', on_esc_key_press)

# FPS calculation variables
prev_time_cam = time.time()
prev_time_radar = time.time()
frame_count_cam = 0
frame_count_radar = 0

# Radar configuration parameters
#############################################################################################################################
start_freq = 77e9
freq_slope = 100e6 / 1e-6
end_freq = 80.9e9
sampling_rate = 7.2e6
num_samples = 256
num_chirp_loops = 32
idle_time = 7e-6
ramp_end_time = 39e-6
adc_valid_start_time = 3e-6
frame_periodicity = 50e-3
speed_of_light = 3e8
num_TX = 3
num_RX = 4
virtual_ant = num_TX * num_RX

# Calculated radar parameters
maximum_beat_freq = min(10e6, 0.8 * sampling_rate)
chirp_time = num_samples / sampling_rate
valid_sweep_bandwidth = chirp_time * freq_slope
chirp_repetition_time = num_TX * (idle_time + ramp_end_time)
carrier_freq = start_freq + freq_slope * adc_valid_start_time + valid_sweep_bandwidth / 2

maximum_range = maximum_beat_freq * speed_of_light / (2 * freq_slope)
range_resolution = speed_of_light / (2 * valid_sweep_bandwidth)
maximum_velocity = speed_of_light / (4 * chirp_repetition_time * carrier_freq)
velocity_resolution = 2 * maximum_velocity / num_chirp_loops

logger.info(f"Radar Parameters: Max Beat Frequency: {maximum_beat_freq}, Chirp Time: {chirp_time}, "
            f"Valid Sweep Bandwidth: {valid_sweep_bandwidth}, Chirp Repetition Time: {chirp_repetition_time}, "
            f"Carrier Frequency: {carrier_freq}, Maximum Range: {maximum_range}, Range Resolution: {range_resolution}, "
            f"Maximum Velocity: {maximum_velocity}, Velocity Resolution: {velocity_resolution}")
#############################################################################################################################

# Initialize radar and data collection variables
dca = None
radar = None
frame_num_in_buffer = 1
num_frames = 10000

# Depth camera configurations
depth_cam_config = {
    'd435': {
        'max_range': 3,
        'fov_azi': 69,
        'fov_ele': 42
    },
    'd455': {
        'max_range': 6,
        'fov_azi': 87,
        'fov_ele': 58
    }
}

# Select depth camera type
root_path = 'dataset/'
camera_type = 'd455'
depth_cam_config = depth_cam_config[camera_type]
logger.info(f"Using {camera_type} camera")

# Application state class
class AppState:
    def __init__(self):
        self.paused = False
        self.decimate = 0

state = AppState()

# Configure depth camera streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# Check if the device has an RGB camera
found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("This demo requires a Depth camera with a Color sensor.")
    exit(0)

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
logger.info(f"Depth Intrinsics: {depth_intrinsics.width}, {depth_intrinsics.height}, {depth_intrinsics.fx}, {depth_intrinsics.fy}, {depth_intrinsics.ppx}, {depth_intrinsics.ppy}")
w, h = depth_intrinsics.width, depth_intrinsics.height

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

def calculate_point_cloud(depth_frame, intrinsics):
    """
    Calculate the 3D point cloud manually from a depth frame using camera intrinsics.
    """
    height, width = depth_frame.shape

    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy

    # Create a mesh grid of pixel coordinates
    x_indices = np.tile(np.arange(width), height).reshape(height, width)
    y_indices = np.repeat(np.arange(height), width).reshape(height, width)

    # Get the depth values for each pixel
    z = depth_frame * depth_scale  # Convert to meters

    # Calculate the 3D coordinates using the pinhole camera model
    x = (x_indices - ppx) * z / fx
    y = (y_indices - ppy) * z / fy

    # Combine X, Y, Z coordinates into a single point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def get_point_cloud():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    logger.info(f"Depth Image Shape: {depth_image.shape}, Color Image Shape: {color_image.shape}")

    # Manually calculate the point cloud
    verts = calculate_point_cloud(depth_image, depth_intrinsics)

    return verts, depth_image, color_image

def cartesian_to_polar(x, y, z, isReverse=True):
    """Convert Cartesian coordinates to polar coordinates."""
    if isReverse:
        x, y, z = -x, -y, z
    range_ = np.sqrt(x**2 + z**2)
    azimuth = np.arctan2(x, z)  # Azimuth remains in radians for polar plot
    return range_, azimuth

if __name__ == '__main__':
    if enable_visualization:
        # Set up the visualization figure and axes
        font_size = 64
        fig = plt.figure(figsize=(100, 25), dpi=20)
        ax_depth_cam = fig.add_subplot(131, polar=True)
        ax_radar = fig.add_subplot(132)
        ax_combined = fig.add_subplot(133)

    # Initialize the radar and DCA1000 data collector
    dca = DCA1000()
    dca.reset_radar()
    dca.reset_fpga()
    time.sleep(1)
    dca_config_file = "cf.json" 
    radar_config_file = "iwr18xx_profile.cfg"
    
    # set the port number according to the actual situation
    radar = TI(cli_loc=cli_loc, data_loc=data_loc, data_baud=921600, config_file=radar_config_file, verbose=True)
    _, _, ADC_PARAMS, _ = DCA1000.read_config(radar_config_file)
    radar.setFrameCfg(num_frames)
    radar.create_read_process(num_frames)
    dca.configure(dca_config_file, radar_config_file)
    logger.debug("Press ENTER to start capture...")
    keyboard.wait('enter')

    # Start the radar collection
    radar.start_read_process()
    dca.stream_start()
    start_time = datetime.datetime.now()
    radar.startSensor()

    while not stop_flag:
        if enable_visualization:
            # Calculate FPS for radar and depth camera visualizations
            current_time_cam = time.time()
            frame_count_cam += 1
            fps_cam = frame_count_cam / (current_time_cam - prev_time_cam)
            ntp_time_cam = datetime.datetime.now().strftime('%H:%M:%S.%f')

            current_time_radar = time.time()
            frame_count_radar += 1
            fps_radar = frame_count_radar / (current_time_radar - prev_time_radar)
            ntp_time_radar = datetime.datetime.now().strftime('%H:%M:%S.%f')

        # Capture depth camera data using the new point cloud calculation
        verts, depth_image, color_image = get_point_cloud()
        if verts is None or depth_image is None or color_image is None:
            continue

        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        range_cam, azimuth_cam = cartesian_to_polar(x, y, z, isReverse=isReverse)

        # Capture and process radar data
        data_buf, loss_rate = dca.fastRead_in_Cpp(frame_num_in_buffer, sortInC=True, isLossReturn=True)
        if loss_rate != 0:
            logger.info("Skip this sample due to loss.")
            continue
        adc_data = np.reshape(data_buf, (-1, ADC_PARAMS['chirps'], ADC_PARAMS['tx'], ADC_PARAMS['rx'], ADC_PARAMS['samples'] // 2, ADC_PARAMS['IQ'], 2))
        adc_data = np.transpose(adc_data, (0, 1, 2, 3, 4, 6, 5))
        adc_data = np.reshape(adc_data, (-1, ADC_PARAMS['chirps'], ADC_PARAMS['tx'], ADC_PARAMS['rx'], ADC_PARAMS['samples'], ADC_PARAMS['IQ']))
        adc_data = (1j * adc_data[..., 0] + adc_data[..., 1]).astype(np.complex64)
        adc_data = adc_data[:, 0, :, :, :]
        processed_data = np.abs(processing(adc_data=adc_data))

        # Align and crop the radar spectrum
        # 14:101 means -43.5 to 43.5 degrees in azimuth
        # 14:142 means 0.6 to 6 meters in range
        processed_data = processed_data[0, 14:101, 14:142]

        # Convert azimuth to degrees
        azimuth_cam_deg = np.degrees(azimuth_cam)

        # Filter depth camera points based on range
        mask = (range_cam >= 0) & (range_cam <= depth_cam_config['max_range'])
        range_cam = range_cam[mask]
        azimuth_cam = azimuth_cam[mask]
        azimuth_cam_deg = azimuth_cam_deg[mask]

        ################################    VISUALIZATION    ################################
        if enable_visualization:
            # Radar visualization
            ax_radar.clear()
            ax_radar.imshow(processed_data, aspect='auto', cmap='viridis')
            ax_radar.set_xlabel("Range (m)", fontsize=font_size)
            ax_radar.set_ylabel("Azimuth (degrees)", fontsize=font_size)
            ax_radar.set_title(f"mmWave Radar Range-Azimuth Spectrum\nNTP Time: {ntp_time_radar} | FPS: {fps_radar:.2f}", fontsize=font_size)
            ax_radar.tick_params(axis='both', which='major', labelsize=font_size)

            # Depth camera visualization
            ax_depth_cam.clear()
            ax_depth_cam.scatter(azimuth_cam, range_cam, s=1, c='red', label='Depth Camera')
            ax_depth_cam.set_title(f"Depth Camera Range-Azimuth Pointcloud\nNTP Time: {ntp_time_cam} | FPS: {fps_cam:.2f}", fontsize=font_size)
            ax_depth_cam.set_xlabel("Range (m)", fontsize=font_size)
            ax_depth_cam.set_ylabel("Azimuth (degrees)", fontsize=font_size)
            ax_depth_cam.set_ylim(0, depth_cam_config['max_range'])
            ax_depth_cam.set_xlim(-depth_cam_config['fov_azi'] / 2 * np.pi / 180, depth_cam_config['fov_azi'] / 2 * np.pi / 180)
            ax_depth_cam.tick_params(axis='both', which='major', labelsize=font_size)

            # Combined radar and depth camera visualization
            ax_combined.clear()
            ax_combined.imshow(processed_data, extent=[0, depth_cam_config['max_range'], -depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2], aspect='auto', cmap='viridis', alpha=0.5)
            ax_combined.scatter(range_cam, azimuth_cam_deg, s=1, c='red', label='Depth Camera')
            ax_combined.set_xlabel("Range (m)", fontsize=font_size)
            ax_combined.set_ylabel("Azimuth (degrees)", fontsize=font_size)
            ax_combined.set_title(f"Combined Range-Azimuth Spectrum\nNTP Time: {ntp_time_cam} | FPS: {fps_cam:.2f}", fontsize=font_size)
            ax_combined.set_xlim(0, depth_cam_config['max_range'])
            ax_combined.set_ylim(-depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2)
            ax_combined.tick_params(axis='both', which='major', labelsize=font_size)

        ################################    SAVE DATA    ################################
        if save_data:

            # Prepare the dictionary to save as .pickle
            ntp_time_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'{volunteer_name}_{test_location}_{ntp_time_filename}'

            data_to_save = {
                'spectrum': processed_data,
                'raw_pointcloud': verts,
                'depth_image': depth_image,
                'color_image': color_image
            }

            # Save the combined image as a PNG binary
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.imshow(processed_data, extent=[0, depth_cam_config['max_range'], -depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2], aspect='auto', cmap='viridis', alpha=0.5)
            ax.scatter(range_cam, azimuth_cam_deg, s=1, c='black', label='Depth Camera')
            ax.set_xlim(0, depth_cam_config['max_range'])
            ax.set_ylim(-depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2)

            # Remove ticks, labels, and titles
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

            # Remove the axis boundaries (spines)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Save the combined image as a transparent PNG binary
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf.seek(0)
            combination_binary = buf.getvalue()
            plt.close(fig)

            # Add the combination image to the pickle data
            data_to_save['combination'] = combination_binary

            # Save the dictionary as a .pickle file
            with open(os.path.join(root_path, f'{filename}.pickle'), 'wb') as f:
                pickle.dump(data_to_save, f)

        # Update the display and pause
        plt.draw()
        plt.pause(0.1)

        if stop_flag:
            break
