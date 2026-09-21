# ExDark Low-Light Object Detection with YOLO

## 1. Project Title

**ExDark Low-Light Object Detection using YOLO**

An object detection project focused on detecting common objects in **low-light and challenging illumination conditions** using YOLO-based deep learning models.

---

## 2. Problem Statement

Object detection systems often perform well in normal lighting conditions but can struggle when images are captured at night, indoors, or under poor illumination.

This project investigates the use of YOLO object detection models to identify and localize objects in **low-light images** from the ExDark dataset.

The goal is to train and evaluate YOLO models and compare their detection performance on challenging low-light images.

---

## 3. Why Computer Vision?

Computer vision is well suited to this problem because the objective is not only to identify what objects are present, but also to determine **where each object is located in the image**.

Object detection provides:

* Object classification
* Bounding-box localization
* Multiple-object detection in a single image
* Real-time or near-real-time inference

YOLO is particularly suitable because it performs object detection using a single-stage architecture and is designed to provide a good balance between detection accuracy and computational efficiency.

---

## 4. Dataset

### Dataset Name

**ExDark — Exclusively Dark Image Dataset**

### Source

The ExDark dataset contains images captured under different low-light conditions and includes multiple object categories.

Dataset source:

**ExDark Dataset:**
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

### Images

The original ExDark dataset contains approximately **7,363 low-light images**.

For this project, the dataset was prepared and converted into a YOLO-compatible object detection format before training.

**Final number of images used:** `[INSERT FINAL NUMBER]`

### Classes

The dataset contains **12 object classes**:

| ID | Class     |
| -: | --------- |
|  0 | Bicycle   |
|  1 | Boat      |
|  2 | Bottle    |
|  3 | Bus       |
|  4 | Car       |
|  5 | Cat       |
|  6 | Chair     |
|  7 | Cup       |
|  8 | Dog       |
|  9 | Motorbike |
| 10 | People    |
| 11 | Table     |

### Annotation Format

The annotations were converted/prepared in **YOLO format**.

Each annotation contains:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized between 0 and 1.

### Dataset Split

The dataset was divided into:

* **Training:** `[INSERT NUMBER / %]`
* **Validation:** `[INSERT NUMBER / %]`
* **Testing:** `[INSERT NUMBER / %]`

The same dataset structure was used for the YOLO experiments.

### Dataset Limitations

The main limitations of the dataset include:

* Images have challenging illumination conditions.
* Some objects are difficult to distinguish because of darkness.
* Objects may be partially hidden or heavily blurred.
* Small objects can be difficult to detect.
* Some classes contain considerably more examples than others.
* Low-light conditions can reduce visual information available to the detector.

These factors make object detection more challenging than detection under normal lighting conditions.

---

## 5. Methodology

The project followed the following pipeline:

```text
ExDark Dataset
       │
       ▼
Dataset Inspection
       │
       ▼
Annotation Preparation
       │
       ▼
YOLO Format Conversion
       │
       ▼
Dataset Validation
       │
       ▼
Train / Validation / Test Split
       │
       ▼
YOLO Model Training
       │
       ├───────────────┐
       ▼               ▼
YOLO11s             YOLO26s
       │               │
       └───────┬───────┘
               ▼
          Model Evaluation
               │
               ▼
       Model Comparison
               │
               ▼
      Predictions on New Images
```

### Data Preprocessing

The preprocessing pipeline included:

1. Inspecting the original image and annotation structure.
2. Checking the available classes and annotations.
3. Converting annotations into YOLO-compatible format.
4. Organizing images and labels into separate directories.
5. Creating the required `data.yaml` configuration file.
6. Validating image-label pairs.
7. Checking class IDs and annotation formatting.
8. Creating training, validation, and testing sets.
9. Verifying that the final dataset could be loaded correctly by Ultralytics YOLO.

No artificial image enhancement was applied as a replacement for the original low-light images. The objective was to evaluate YOLO directly on the challenging illumination conditions represented by ExDark.

---

## 6. Model

Two YOLO models were trained and compared:

### YOLO11s

The YOLO11 small model was selected because it provides a relatively lightweight architecture while maintaining strong object-detection capabilities.

### YOLO26s

YOLO26s was also evaluated as a newer YOLO architecture and provides a second model for comparison.

Both models were trained using the same dataset preparation and experimental conditions to make the comparison more meaningful.

### Training Configuration

The experiments were intentionally limited to **10 epochs**.

Main settings:

```text
Epochs: 10
Task: Object Detection
Dataset: ExDark
Image size: [INSERT IMAGE SIZE]
Device: [CPU / GPU]
Models: YOLO11s and YOLO26s
```

The relatively small number of epochs was chosen because the project was developed in a resource-constrained environment and the goal was to establish a baseline comparison.

---

## 7.  Training Results

Training metrics were monitored throughout the 10 epochs, including:

* Box loss
* Classification loss
* DFL loss
* Precision
* Recall
* mAP@50
* mAP@50-95

Training curves are available in:

```text
results/training_results/
```

### Confusion Matrix

The confusion matrix is included to analyze classification errors between object categories.

Example:

```text
results/confusion_matrix/confusion_matrix.png
```

The confusion matrix helps identify classes that are frequently confused with one another, which can be particularly important in low-light images.

### Sample Predictions

Example predictions are included in:

```text
results/predictions/
```

Each prediction shows:

* Detected object
* Bounding box
* Confidence score
* Predicted class

---

## 8. Error Analysis

Low-light object detection presents several sources of error.

Typical errors observed during evaluation include:

### Missed Objects

Some objects may not be detected because they are too dark, too small, partially hidden, or visually similar to their surroundings.

### False Positives

The model may interpret background regions or visual patterns as objects.

### Class Confusion

Objects with similar visual characteristics can be incorrectly classified, particularly when illumination removes important visual details.

### Small Objects

Small objects occupy fewer pixels and therefore contain less information for the detector.

### Difficult Illumination

Very dark regions can make object boundaries difficult to identify.

Example incorrect predictions should be added to:

```text
results/error_analysis/
```

Each example should include a short explanation of what went wrong.

---

## 9. Demo

The project can be demonstrated by running the trained YOLO model on previously unseen images.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("example.jpg")

for result in results:
    result.show()
```


## 10. Installation & Usage

### Requirements

Python 3.x and the following packages are required:

```text
ultralytics
torch
opencv-python
matplotlib
pandas
numpy
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### Dataset

Download the ExDark dataset from the official dataset source and prepare it according to the directory structure described in `data.yaml`.

### Training

Example YOLO training command:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640
)
```

The same procedure can be used with the second YOLO architecture.

### Evaluation

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(data="data.yaml")
```

### Prediction

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test_image.jpg")

for result in results:
    result.show()
```

---

## 11. Future Improvements

Several improvements could be explored in future work:

* Train for more than 10 epochs.
* Perform hyperparameter optimization.
* Increase the amount of training data.
* Address class imbalance.
* Apply appropriate low-light image enhancement techniques.
* Compare additional YOLO architectures.
* Experiment with different image resolutions.
* Apply data augmentation.
* Perform more extensive error analysis.
* Evaluate the models on additional low-light datasets.
* Develop a web-based or real-time detection demo.
* Optimize the final model for faster inference.

---

## 12. Team / Author

### Author

**Mike Mazraany**

AI / Data Science Student
Lebanon

This project was developed as part of the **AI Builders Bootcamp**.

---

## Conclusion

This project demonstrates an end-to-end computer vision workflow for detecting objects under challenging low-light conditions.

The workflow covers:

**Dataset preparation → annotation processing → validation → YOLO training → evaluation → model comparison → error analysis → inference**

The comparison between YOLO11s and YOLO26s provides a practical evaluation of two YOLO architectures under the same experimental conditions.
