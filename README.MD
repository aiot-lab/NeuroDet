# Unfolding Target Detection with State Space Model
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains the open-source code for the paper: **NeuroDet: Unfolding Target Detection with State Space Model**. [[paper]](./preprint.pdf)

## Paper Abstract

Target detection is a fundamental task in radar sensing, serving as the precursor to any further processing for various applications. Numerous detection algorithms have been proposed. Classical methods based on signal processing, e.g., the most widely used CFAR, are challenging to tune and sensitive to environmental conditions. Deep learning-based methods can be more accurate and robust, yet usually lack interpretability and physical relevance. In this paper, we introduce a novel method that combines signal processing and deep learning by unfolding the CFAR detector with a state space model architecture. By reserving the CFAR pipeline yet turning its sophisticated configurations into trainable parameters, our method achieves high detection performance without manual parameter tuning, while preserving model interpretability. We implement a lightweight model of only 260K parameters and conduct real-world experiments for human target detection using FMCW radars. The results highlight the remarkable performance of the proposed method, outperforming CFAR and its variants by 10 times in detection rate and false alarm rate.

## Repository Structure

```bash
.
├── 3d_printed_case
│   ├── 3d_printed_case.dwg
│   └── 3d_printed_case.stl
├── args.py
├── data_collection
│   ├── cf.json
│   ├── data_collection.py
│   ├── data_collection.sh
│   ├── iwr18xx_profile.cfg
│   ├── postprocessing.py
│   └── radar_processing.py
├── mmwave
│   └── dataloader
│       └── adc.py
├── README.MD
├── requirements.txt
├── train_model.py
├── train.sh
└── utils
    ├── dataset.py
    ├── loss.py
    └── model.py
```

## Setup Instructions

1. **Clone the `fpga_udp` repository**:
   ```bash
   git clone https://github.com/username/fpga_udp.git
   ```

2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure

- **`3d_printed_case`**: Contains `.dwg` and `.stl` files for the 3D-printed case, which can be downloaded and printed directly.
- **`data_collection`**: Contains configurations for mmWave radar and pre-processing functions.
- **`utils`**: Includes Python files for the dataset, model, and loss functions.
- **`mmwave`**: Contains an override function `fastRead_in_Cpp` for the original folders.

## Usage Instructions

### Data Collection

1. **Connect all devices** correctly to the PC.
2. **Run the data collection script** with appropriate settings (e.g., user and data ports, static Ethernet IP of DCA1000EVM):
    ```bash
    cd data_collection
    bash data_collection.sh
    ```
3. The saved data will consist of:
   - Spectrum of mmWave radar
   - Raw point cloud data
   - Depth image
   - RGB image from the depth camera

4. **Download YOLOv8 and SAM weights**:
   - YOLOv8 weights: [Download here](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
   - SAM weights: [Download here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

5. **Set the paths in `postprocessing.py`** and run the script.

### Model Training

1. Ensure that each element in `DATASET_PATHS` within `train.sh` has two subfolders: `train` and `test`. Each subfolder should contain `.pickle` files. Additionally, the elements in `CALIBRATION_PATHS` should include the calibration `.pickle` files.

2. **Run the training script** with the appropriate settings:
   ```bash
   bash train.sh
   ```
