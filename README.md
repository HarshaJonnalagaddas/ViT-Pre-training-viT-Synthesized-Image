# Segment Anything Using Vision Transformers (ViT)
This repository demonstrates the use of Meta's Segment Anything Model (SAM) with Vision Transformers (ViT) for image segmentation. It includes step-by-step instructions to set up the environment, download necessary files, execute the segmentation script, and visualize the results.

---

## Table of Contents
1. [Objective](#objective)
2. [Features](#features)
3. [Installation](#installation)
4. [Setup](#setup)
5. [Steps Performed](#steps-performed)
6. [Execution](#execution)
7. [Results](#results)
8. [Challenges and Resolutions](#challenges-and-resolutions)
9. [References](#references)

---

## Objective

To implement and demonstrate the functionality of Meta's Segment Anything Model (SAM) using the pre-trained `vit_h.pth` checkpoint for accurate and efficient image segmentation tasks.

---

## Features

- **Pre-trained SAM Model Integration**: Utilizes the `vit_h` variant for advanced segmentation.
- **Interactive Prompts**: Accepts user-defined points to generate segmentation masks.
- **Multi-mask Generation**: Outputs multiple mask options for a single prompt.
- **Visualization**: Displays the segmentation results using Matplotlib.

---

## Installation

### Prerequisites

- **Python Version**: Python 3.8 or higher
- **Hardware**: GPU recommended for faster execution (optional)

### Required Libraries

Install the necessary Python libraries using the command below:

```bash
pip install torch torchvision numpy opencv-python matplotlib
```

---

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/segment-anything-vit.git
   cd segment-anything-vit
   ```

2. **Download the SAM Checkpoint**
   - Visit [Meta's Segment Anything GitHub page](https://github.com/facebookresearch/segment-anything).
   - Download the `vit_h.pth` checkpoint file and place it in the project directory.

3. **Update Configuration**
   Open the `sam_demo.py` file and update the following variables:
   - `sam_checkpoint`: Path to the downloaded checkpoint file (e.g., `path/to/vit_h.pth`).
   - `image_path`: Path to the input image file for segmentation.

---

## Steps Performed

### 1. Model Setup
- Selected the `vit_h` variant of SAM, leveraging its high-resolution image processing capabilities.
- Loaded the pre-trained checkpoint file (`vit_h.pth`).
- Ensured compatibility with GPU (`cuda`) if available, or defaulted to CPU.

### 2. Image Preprocessing
- Loaded the input image using OpenCV.
- Converted the image from BGR format (default in OpenCV) to RGB, aligning with the model's input requirements.

### 3. Interactive Segmentation
- Defined a user-specified point-based prompt for segmentation. For example:
  - `input_point = np.array([[500, 375]])`
  - `input_label = np.array([1])`
- Generated segmentation masks based on the provided prompt.

### 4. Visualization
- Visualized the resulting segmentation mask using Matplotlib.
- Allowed for user inspection of the generated masks.

---

## Execution

Run the segmentation script with the following command:

```bash
python sam_demo.py
```

Upon execution, the script will:
1. Load the SAM model and the input image.
2. Apply the point-based segmentation prompt.
3. Generate and display the segmentation masks.

---

## Results

The project successfully performs segmentation based on user-defined prompts, producing detailed masks of the target object in the image.

### Example Output

Here is an example of the segmentation mask generated:
![image](https://github.com/user-attachments/assets/9e2db0b2-faa5-4e3f-8588-c09cc2c44b7b)
![image](https://github.com/user-attachments/assets/a85e455c-8cc7-4128-bfbb-a6e9ea5cda72)


---

## Challenges and Resolutions

### Challenges

1. **Path Management**
   - Initial errors due to incorrect file paths for the model checkpoint and input image.
   - **Resolution**: Verified paths and used absolute paths for reliability.

2. **Device Compatibility**
   - GPU compatibility issues on certain systems.
   - **Resolution**: Implemented a fallback mechanism to CPU if GPU is unavailable.

### Lessons Learned

- The importance of precise path and dependency management in deep learning workflows.
- The need for robust testing across different hardware configurations.

---

## References

1. [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
3. [OpenCV Documentation](https://docs.opencv.org/)

---

Feel free to contribute, raise issues, or suggest improvements. This repository aims to serve as a starting point for further exploration of SAM and ViT integration.
