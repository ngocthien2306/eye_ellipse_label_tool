# Pupil Annotation Tool

A comprehensive tool for annotating, processing, and augmenting eye pupil detection datasets for computer vision applications.

## Features

- **Annotation**: Create precise pupil annotations with segmentation points and ellipse fitting
- **Data Splitting**: Split datasets into train/validation/test sets with customizable ratios
- **Eye Region Cropping**: Automatically crop images to the eye region
- **Data Augmentation**: Generate augmented datasets with various transformations

## Screenshots

### Annotation Screen
![Annotation Screen](assets/annotation_screen.png)

The annotation screen allows you to:
- Load image folders 
- Mark segmentation points around the pupil
- Automatically fit ellipse to points
- Save annotations in YOLO format
- Process labeled images to a standardized format

### Data Splitting Screen
![Split Data Screen](assets/split_data_screen.png)

The split screen provides functionality to:
- Load annotated datasets
- Set train/validation/test ratios
- Enable optional eye region cropping
- Preview annotations with visualization
- Process splits with consistent directory structure

### Augmentation Screen
![Augmentation Screen](assets/augmentation_screen.png)

The augmentation screen offers:
- Multiple transformation options (flips, rotation, blur, crop, noise)
- Preview of augmentation effects
- Batch processing of dataset images
- Preservation of annotation integrity during transformations

## Directory Structure

The tool expects and generates the following directory structure:

```
dataset/
├── images/            # Original images
├── labels/            # Segmentation annotations (YOLO format)
├── labels_ellipse/    # Ellipse annotations (YOLO format)
├── data/              # Processed dataset with consistent naming
│   ├── images/  
│   ├── labels/
│   └── labels_ellipse/
└── split/             # Train/validation/test splits
    ├── train/
    │   ├── images/
    │   ├── labels/
    │   └── labels_ellipse/
    ├── valid/
    │   ├── images/
    │   ├── labels/
    │   └── labels_ellipse/
    └── test/
        ├── images/
        ├── labels/
        └── labels_ellipse/
```

## Annotation Format

### Segmentation Format (YOLO)
```
class_id x1 y1 x2 y2 x3 y3 ...
```
Where:
- `class_id`: Integer class identifier (0 for pupil)
- `x1 y1 ... xn yn`: Normalized coordinates (0-1) of segmentation points

### Ellipse Format (YOLO)
```
class_id center_x center_y axes_x axes_y angle
```
Where:
- `class_id`: Integer class identifier (0 for pupil)
- `center_x center_y`: Normalized coordinates (0-1) of ellipse center
- `axes_x axes_y`: Normalized half-lengths of the ellipse axes
- `angle`: Rotation angle in degrees

## Requirements

- Python 3.8+
- PyQt5
- OpenCV-Python
- NumPy
- Albumentations (for augmentation)

## Installation

1. Clone the repository:
```
git clone https://github.com/ngocthien2306/eye_ellipse_label_tool.git
cd eye_ellipse_label_tool
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python main.py
```

## Usage Guide

### Annotation Workflow

1. Click "Select Image Folder" on the Annotation tab
2. Click on the pupil boundary to add segmentation points (minimum 5 points recommended)
3. Click "Fit/Unfit Ellipse" to automatically fit an ellipse to your points
4. Click "Save YOLO Labels" to save the annotations
5. Use "Process Labeled Images" to prepare the dataset for training

### Data Splitting Workflow

1. Click "Load Data Folder" on the Split tab
2. Adjust the split ratios as needed (default: 70% train, 15% validation, 15% test)
3. Optionally enable crop for eye region
4. Click "Split Data" to generate the train/validation/test sets

### Augmentation Workflow

1. Click "Select Data Folder" on the Augmentation tab
2. Choose which dataset to augment (train/validation/test)
3. Configure transformation options
4. Click "Generate Preview" to see example augmentations
5. Set "Augmentations per image" value
6. Click "Generate Augmented Data" to process the entire dataset

## Contribution Guidelines

Contributions to improve the Pupil Annotation Tool are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Coding Standards

- Follow PEP 8 guidelines for Python code
- Use descriptive variable and function names
- Add comments for complex logic
- Write unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Pupil Annotation Tool Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Credits

Developed for pupil tracking and eye detection applications.