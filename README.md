# MammoDetect: Breast Cancer Detection Using Convolutional Neural Networks (CNN)

## Project Overview

MammoDetect is a machine learning project designed to detect breast cancer from mammography images using CNN models. It utilizes the **CBIS-DDSM** dataset, a comprehensive collection of breast images, and aims to classify images based on cancerous and non-cancerous findings.

---

### Table of Contents

1. [Dataset Description](#dataset-description)
2. [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Example Predictions](#example-predictions)
6. [Usage Instructions](#usage-instructions)
7. [Contributing](#contributing)

---

## Dataset Description

The **CBIS-DDSM** dataset used in MammoDetect includes high-resolution mammography images with key details as follows:

- **Number of Studies:** 6,775
- **Number of Images:** 10,239
- **Participants:** 1,566 unique individuals
- **Modality:** Mammography (MG)
- **Image Size:** 6 GB in JPEG format

---

## Data Preprocessing and Augmentation

MammoDetect preprocesses images by resizing, normalizing, and applying augmentations such as random rotation, affine transformation, and horizontal flipping.

### Sample Images

- **Cropped Image**
  - ![Cropped Image](images/sample_cropped.png)
- **Full Mammogram Image**
  - ![Full Mammogram Image](images/sample_full_mammogram.png)
- **ROI Mask Image**
  - ![ROI Mask Image](images/sample_roi_mask.png)

---

## Model Architecture

The custom CNN in MammoDetect consists of four convolutional layers, batch normalization, max pooling, and dropout layers, designed to classify mammography images into **Cancerous** and **Non-Cancerous** categories.

---

## Training and Evaluation

The model in MammoDetect was trained for 25 epochs with cross-entropy loss and an Adam optimizer, evaluating performance through accuracy and loss metrics.

### Model Performance

- **Training Accuracy:** Example values
- **Validation Accuracy:** Example values

### Accuracy and Loss Plot

- **Training & Validation Accuracy**

  - ![Accuracy Plot](images/accuracy_plot.png)

- **Training & Validation Loss**
  - ![Loss Plot](images/loss_plot.png)

---

## Example Predictions

Here are example predictions made by MammoDetect on test images:

- **Prediction on Cancerous Image**
  - ![Cancerous Image](images/can.png)
- **Prediction on Non-Cancerous Image**
  - ![Non-Cancerous Image](images/non-can.png)

---

## Usage Instructions

### Prerequisites

- Python 3.x
- Required libraries: `torch`, `PIL`, `pandas`, `matplotlib`, `plotly`, `seaborn`

### Steps

1. Clone this repository.
2. Download the **CBIS-DDSM** dataset using the `kagglehub` library or other methods.
3. Preprocess and train the model using the provided script.
4. Use the inference function to predict on new images.

### Inference Example

```python
from model import load_model, infer_image

model = load_model('CNN_model.pth')
predicted_label, confidence = infer_image(model, 'path/to/image.jpg')
print(f'Prediction: {predicted_label}, Confidence: {confidence}')
```

---

## Contributing

For contributions, please open a pull request or submit an issue for discussion.
