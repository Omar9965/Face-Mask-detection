# Face Mask Detection

This project utilizes deep learning to detect whether a person is wearing a face mask. The implementation is based on TensorFlow/Keras and processes image datasets for classification.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example Output](#example-output)

---

## About the Project
The **Face Mask Detection** project aims to:
- **Classify images** as "with mask" or "without mask."
- **Utilize CNN-based deep learning models** to enhance accuracy.
- **Improve public health monitoring** by providing an automated face mask detection tool.

The dataset consists of labeled images stored in directories, and the model is trained using TensorFlow/Keras.

---

## Features
- **Image Preprocessing**: Reads and processes images for model training.
- **Mask Classification**: Detects whether a person is wearing a face mask.
- **Deep Learning Model**: Leverages Convolutional Neural Networks (CNNs) for classification.
- **Data Augmentation**: Improves model generalization through transformations.

---

## Getting Started

Follow these steps to set up and run the project locally:

### Prerequisites
- Python 3.8+
- TensorFlow/Keras
- OpenCV & PIL for image handling
- Scikit-learn for dataset splitting

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd face-mask-detection
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is available in the specified directory:
   - `data/with_mask/`
   - `data/without_mask/`

---

## Usage

Run the notebook to train and test the model:
```bash
jupyter notebook face-mask-detection.ipynb
```

After training, you can use the model to predict mask usage on new images.

---

## Project Structure
The project files are organized as follows:
```
face-mask-detection/
├── data/
│   ├── with_mask/            # Images with masks
│   ├── without_mask/         # Images without masks
├── model/
│   ├── trained_model.h5      # Saved Keras model
├── face-mask-detection.ipynb # Jupyter Notebook for training and testing
├── requirements.txt          # List of required dependencies
└── README.md                 # Project documentation
```

