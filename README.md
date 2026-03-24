# 🧠 Alzheimer’s MRI Classification using CNN

## 📌 Overview

This project focuses on detecting Alzheimer’s disease stages using MRI brain images and deep learning techniques. A Convolutional Neural Network (CNN) is trained to classify MRI scans into different stages of Alzheimer’s disease.

The system also includes **model evaluation** and **Grad-CAM visualization** to explain predictions, making it suitable for medical AI applications.

---

## 🎯 Objectives

* Classify MRI brain images into Alzheimer’s stages
* Build and train a CNN model using TensorFlow/Keras
* Evaluate model performance using confusion matrix and metrics
* Visualize model decision-making using Grad-CAM

---

## 🧪 Dataset

* **Dataset:** Augmented Alzheimer MRI Dataset
* **Classes:**

  * NonDemented
  * VeryMildDemented
  * MildDemented
  * ModerateDemented

📥 Download dataset from Kaggle:
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

📁 After downloading, place it like:

```
alzheimer-mri-classification/
│
├── AugmentedAlzheimerDataset/
```

---

## 🏗️ Project Structure

```
alzheimer-mri-classification
│
├── cnn_model.py        # CNN architecture
├── train.py            # Model training
├── predict.py          # Single image prediction
├── evaluation.py       # Confusion matrix & metrics
├── gradcam.py          # Grad-CAM visualization
├── demo.py             # Combined demo (prediction + heatmap + evaluation)
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### 1️⃣ Create virtual environment

```bash
python3.11 -m venv tf-env
source tf-env/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Train Model

```bash
python train.py
```

---

### 🔍 Predict on MRI Image

```bash
python predict.py
```

---

### 📊 Evaluate Model

```bash
python evaluation.py
```

Outputs:

* Confusion Matrix
* Classification Report

---

### 🔥 Grad-CAM Visualization

```bash
python gradcam.py
```

Outputs:

* MRI image with heatmap showing important regions

---

### 🧪 Full Demo (Recommended)

```bash
python demo.py
```

Displays:

* Prediction + MRI image
* Grad-CAM heatmap
* Confusion Matrix

---

## 🧠 Model Details

* Model Type: Convolutional Neural Network (CNN)
* Input Size: 128 × 128 × 3
* Activation: ReLU, Softmax
* Optimizer: Adam
* Loss Function: Categorical Crossentropy

---

## 📈 Results

* Training Accuracy: ~98%
* Validation Accuracy: ~88%
* Evaluation Accuracy: ~97%

The model performs well across all classes with high precision and recall.

---

## 🔬 Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which regions of the MRI image influence the model’s prediction.

This improves:

* Model interpretability
* Trust in medical AI systems

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📌 Key Features

✔ CNN-based MRI classification
✔ Multi-class Alzheimer detection
✔ Confusion matrix evaluation
✔ Grad-CAM explainability
✔ End-to-end ML pipeline

---

## ⚠️ Notes

* Dataset is not included due to size limitations
* Ensure correct dataset path before running scripts
* Use Python 3.10 or 3.11 for TensorFlow compatibility

---

## 👨‍💻 Author

Rohit

---

## 📚 Future Work

* Improve model using transfer learning (ResNet, VGG)
* Build web app for real-time MRI upload and prediction
* Use multimodal data (MRI + clinical data)

---

## ⭐ Acknowledgements

* Kaggle dataset contributors
* TensorFlow & Keras documentation
