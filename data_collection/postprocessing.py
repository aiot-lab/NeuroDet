
#############################################
# Author: Luca Yu
# Date: 2024-08-12
# Description: This file is used to postprocess the raw dataset.
#############################################

import os
import pickle
import io

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


class postprocessing(Dataset):
    def __init__(self, raw_dataset_path, yolo_model, sam_model, sam_predictor, save_processed_folder_path, save_processed_save_visualization_folder):
        self.data_list = []
        self.yolo_model = yolo_model
        self.sam_model = sam_model
        self.sam_predictor = sam_predictor
        self.save_processed_folder_path = save_processed_folder_path
        self.save_processed_save_visualization_folder = save_processed_save_visualization_folder
        # Read all *.pickle absolute file paths under raw_dataset_path
        for root, dirs, files in os.walk(raw_dataset_path):
            for file in files:
                if file.endswith('.pickle'):
                    self.data_list.append(os.path.join(root, file))
        logger.info(f"Total {len(self.data_list)} files found in {raw_dataset_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_path = self.data_list[idx]
        spectrum, raw_pointcloud, depth_image, color_image, combination = self.load_pickle(sample_path)
        self.visualize_data(sample_path, spectrum, raw_pointcloud, depth_image, color_image, combination, self.save_processed_folder_path, self.save_processed_save_visualization_folder)
        return 0

    def load_pickle(self, file_path):
        """Load the pickle file from the given path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        spectrum = data.get('spectrum')
        raw_pointcloud = data.get('raw_pointcloud')
        depth_image = data.get('depth_image')
        color_image = data.get('color_image')
        combination = data.get('combination')

        return spectrum, raw_pointcloud, depth_image, color_image, combination

    def cartesian_to_polar(self, x, y, z):
        """Convert Cartesian coordinates to polar coordinates for point cloud."""
        range_ = np.sqrt(x**2 + z**2)
        azimuth = np.arctan2(-x, z)  # Azimuth remains in radians for polar plot

        mask = (range_ >= 0) & (range_ <= 6)  # Adjust based on your setup's max range
        range_, azimuth = range_[mask], azimuth[mask]

        azimuth_deg = np.degrees(azimuth)
        return range_, azimuth, azimuth_deg

    def yolov8_detection(self, image):
        results = self.yolo_model(image, stream=True)
        person_boxes = []
        for result in results:
            for box in result.boxes:
                if box.cls.item() == 0:  # 0 is the class index for 'person'
                    # Convert bounding box coordinates to integers
                    person_boxes.append([int(coord) for coord in box.xyxy.tolist()[0]])

        return person_boxes

    def segment_person(self, image, bboxes):
        # Ensure the image is loaded correctly
        if image is None:
            raise ValueError("The image could not be loaded. Check the image path and format.")

        # Set the image for SAM predictor
        self.sam_predictor.set_image(image)

        # Perform segmentation for all bounding boxes
        masks_combined = None
        for bbox in bboxes:
            input_boxes = torch.tensor([bbox], device=self.sam_predictor.device)  # Convert bbox to tensor
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            person_mask = masks[0].squeeze().cpu().numpy()

            if masks_combined is None:
                masks_combined = person_mask
            else:
                masks_combined = np.maximum(masks_combined, person_mask)  # Combine masks from multiple persons

        return masks_combined


    def calculate_point_cloud(self, depth_frame, intrinsics, depth_scale):
        height, width = depth_frame.shape
        fx, fy = intrinsics['fx'], intrinsics['fy']
        ppx, ppy = intrinsics['ppx'], intrinsics['ppy']
        x_indices = np.tile(np.arange(width), height).reshape(height, width)
        y_indices = np.repeat(np.arange(height), width).reshape(height, width)
        z = depth_frame * depth_scale
        x = (x_indices - ppx) * z / fx
        y = (y_indices - ppy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return points

    def apply_mask(self, image, mask):
        masked_image = image.copy()
        masked_image[mask == 0] = 0  # Set background to black
        return masked_image

    def filter_point_cloud(self, verts, mask, depth_image):
        height, width = depth_image.shape
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        mask_flat = mask.flatten()
        verts_filtered = verts[mask_flat == 1]
        verts_background = verts[mask_flat == 0]  # Get the background point cloud
        return verts_filtered, verts_background

    def visualize_data(self, file_path, spectrum, raw_pointcloud, depth_image, color_image, combination, save_processed_folder_path, save_processed_save_visualization_folder):
        fig = plt.figure(figsize=(60, 30), constrained_layout=True)

        ax1 = fig.add_subplot(2, 6, 1)
        ax2 = fig.add_subplot(2, 6, 2, polar=True)
        ax3 = fig.add_subplot(2, 6, 3)
        ax4 = fig.add_subplot(2, 6, 4)
        ax5 = fig.add_subplot(2, 6, 5)
        ax6 = fig.add_subplot(2, 6, 6)
        ax7 = fig.add_subplot(2, 6, 7)
        ax8 = fig.add_subplot(2, 6, 8)
        ax9 = fig.add_subplot(2, 6, 9, polar=True)
        ax10 = fig.add_subplot(2, 6, 10)
        ax11 = fig.add_subplot(2, 6, 11)
        ax12 = fig.add_subplot(2, 6, 12)

        # ax1: Radar Spectrum
        ax1.imshow(spectrum, aspect='auto', cmap='viridis')

        # ax2: Raw Point Cloud (Polar)
        x, y, z = raw_pointcloud[:, 0], raw_pointcloud[:, 1], raw_pointcloud[:, 2]
        range_, azimuth, _ = self.cartesian_to_polar(x, y, z)
        
        ax2.scatter(azimuth, range_, s=1, c='red')
        ax2.set_ylim(0, 6)  # Adjust range limits based on your setup
        ax2.set_xlim(-87 / 2 * np.pi / 180, 87 / 2 * np.pi / 180)

        # ax3: Depth Image
        flipped_depth_image = cv2.flip(depth_image, 0)
        ax3.imshow(flipped_depth_image, cmap='gray')

        # ax4: Color Image
        flipped_color_image = cv2.flip(color_image, 0)
        ax4.imshow(cv2.cvtColor(flipped_color_image, cv2.COLOR_BGR2RGB))

        # ax5: Combined Image
        combination_img = plt.imread(io.BytesIO(combination), format='png')
        ax5.imshow(combination_img)

        bbox = self.yolov8_detection(color_image)
        if not bbox:
            logger.info(f"No person detected in {file_path}, skipping this sample.")
            return None  # Skip this sample
        
        person_mask = self.segment_person(color_image, bbox)

        # Load a pre-recorded depth image
        depth_image = depth_image.astype(np.float32)  # Convert to float32 for scaling

        # Manually set the depth camera intrinsics and scale
        depth_intrinsics = {
            'width': 640,
            'height': 480,
            'fx': 387.7923278808594,
            'fy': 387.7923278808594,
            'ppx': 322.8212890625,
            'ppy': 240.0816650390625
        }
        depth_scale = 0.0010000000474974513

        # Calculate the 3D point cloud from the depth image
        verts = self.calculate_point_cloud(depth_image, depth_intrinsics, depth_scale)

        # Apply mask to color and depth images
        masked_color_image = self.apply_mask(color_image, person_mask)
        masked_depth_image = self.apply_mask(depth_image, person_mask)
        
        # Flip the depth image and color image
        masked_color_image = cv2.flip(masked_color_image, 0)
        masked_depth_image = cv2.flip(masked_depth_image, 0)

        # Filter the point cloud using the mask
        verts_filtered, verts_background = self.filter_point_cloud(verts, person_mask, depth_image)

        # convert verts to polar coordinates
        x_filtered = verts_filtered[:, 0]
        y_filtered = verts_filtered[:, 1]
        z_filtered = verts_filtered[:, 2]
        range_filtered, azimuth_filtered, azimuth_filtered_deg = self.cartesian_to_polar(x_filtered, y_filtered, z_filtered)

        x_background = verts_background[:, 0]
        y_background = verts_background[:, 1]
        z_background = verts_background[:, 2]
        range_background, azimuth_background, azimuth_background_deg = self.cartesian_to_polar(x_background, y_background, z_background)


        # ax6: Masked Color Image
        ax6.imshow(cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2RGB))

        # ax7: Masked Depth Image
        ax7.imshow(masked_depth_image, cmap='gray')

        # ax8: Range-Azimuth Plot
        ax8.scatter(range_background, azimuth_background_deg, s=1, c='blue', label='Background')
        ax8.scatter(range_filtered, azimuth_filtered_deg, s=1, c='red', label='Person')
        ax8.set_xlim(0, 6)  # Adjust range limits based on your setup
        ax8.set_ylim(-87/2, 87/2)
        ax8.legend(loc='upper right')

        # ax9: Range-Azimuth Plot (Polar)
        ax9.scatter(azimuth_background, range_background, s=1, c='blue', label='Background')
        ax9.scatter(azimuth_filtered, range_filtered, s=1, c='red', label='Person')
        ax9.set_ylim(0, 6)  # Adjust range limits based on your setup
        ax9.set_xlim(-87 / 2 * np.pi / 180, 87 / 2 * np.pi / 180)
        ax9.legend(loc='upper right')

        # ax10: Spectrum Plot (with Overlaid Point Cloud)
        ax10.imshow(spectrum, aspect='auto', cmap='viridis', extent=[0, 6, -87/2, 87/2], alpha=0.5)
        ax10.scatter(range_background, azimuth_background_deg, s=1, c='blue', label='Background', alpha=0.5)
        ax10.scatter(range_filtered, azimuth_filtered_deg, s=1, c='red', label='Person', alpha=0.5)
        ax10.set_xlim(0, 6)  # Adjust azimuth limits based on your FOV
        ax10.set_ylim(-87/2, 87/2)  # Adjust range limits based on your setup

        # get the name of the file
        filename = os.path.basename(file_path)
        logger.info(f"filename: {filename}")

        # save_pointcloud_as_npy(range_background, azimuth_background_deg, range_filtered, azimuth_filtered_deg, spectrum.shape, output_path_png, output_path_npy)

        spectrum_shape = spectrum.shape
        result = np.zeros(spectrum_shape)

        # Scale the azimuth and range to match the spectrum resolution
        azimuth_indices_bg = np.clip(((azimuth_background_deg + 87/2) / 87 * spectrum_shape[0]).astype(int), 0, spectrum_shape[0] - 1)
        range_indices_bg = np.clip((range_background / 6 * spectrum_shape[1]).astype(int), 0, spectrum_shape[1] - 1)
        
        azimuth_indices_fg = np.clip(((azimuth_filtered_deg + 87/2) / 87 * spectrum_shape[0]).astype(int), 0, spectrum_shape[0] - 1)
        range_indices_fg = np.clip((range_filtered / 6 * spectrum_shape[1]).astype(int), 0, spectrum_shape[1] - 1)

        # Assign values to the point cloud in the result matrix
        result[azimuth_indices_bg, range_indices_bg] = 1  # Background points valued as 1
        result[azimuth_indices_fg, range_indices_fg] = 2  # Person points valued as 2

        # ax11: Person and Background Point Cloud
        ax11.imshow(result, cmap='gray', origin='lower')

        # ax12: Spectrum Plot (with Overlaid Point Cloud)
        spectrum = np.flipud(spectrum)
        ax12.imshow(spectrum, aspect='auto', cmap='viridis', alpha=0.5)
        ax12.imshow(result, origin='lower', alpha=0.5)

        plt.tight_layout()
        plt.show()

        save_processed_path =  save_processed_save_visualization_folder + '/' + filename.replace('.pickle', '.png')
        plt.savefig(save_processed_path)

        result_processed = {
            'spectrum': spectrum,
            'pointcloud': result
        }

        # save the result as a pickle file
        save_processed_path = save_processed_folder_path + '/' + filename

        with open(save_processed_path, 'wb') as f:
            pickle.dump(result_processed, f)

        plt.close(fig)

if __name__ == "__main__":
    raw_dataset_path = '/path/to/your/raw/dataset'
    save_processed_folder_path = '/path/to/your/processed/dataset'
    save_processed_save_visualization_folder = '/path/to/your/processed/visualization'

    # Load the YOLO and SAM models only once
    yolo_model_path = '/path/to/your/yolov8/weights'
    sam_checkpoint = '/path/to/your/sam/weights'
    model_type = "vit_h"

    # Initialize the models
    yolo_model = YOLO(yolo_model_path)
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam_model)

    # Initialize the dataset and dataloader
    dataset = postprocessing(raw_dataset_path, yolo_model, sam_model, sam_predictor, save_processed_folder_path, save_processed_save_visualization_folder)
    dataloader = DataLoader(dataset, batch_size=12, num_workers=0, shuffle=False)

    # Use tqdm to show the progress
    for i, data in enumerate(tqdm(dataloader)):
        pass

