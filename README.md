# 🫁 CT Image-Based Lung Cancer Prediction System

## 📌 Overview

This project implements a complete machine learning and deep learning pipeline for detecting lung cancer from CT scan images.
It uses traditional models (KNN, SVM) and a Convolutional Neural Network (CNN) for accurate classification.

The system is designed with a modular structure and includes a Streamlit-based web interface for real-time prediction.

---

## 🧠 Project Workflow

```
data/raw → preprocessing → model training → model saving → deployment (Streamlit)
```

### Step-by-step Flow:

1. Load CT scan images from dataset
2. Preprocess images (resize, grayscale, normalize)
3. Split data into training and testing sets
4. Train models (KNN, SVM, CNN)
5. Evaluate performance (accuracy, confusion matrix)
6. Save trained CNN model
7. Use Streamlit app for prediction

---

## 📂 Project Structure

```
lung_cancer_project/
│
├── data/
│   ├── raw/                # Original dataset (cancer / normal)
│   ├── processed/          # Preprocessed images
│   ├── train/              # Training images (optional)
│   └── test/               # Testing images (optional)
│
├── models/
│   └── cnn_model.h5        # Trained CNN model
│
├── notebooks/
│   └── experiments.ipynb   # Main training & experimentation file
│
├── src/
│   ├── models.py           # CNN architecture
│   ├── preprocessing.py    # Image preprocessing logic
│   └── utils.py            # Helper functions
│
├── app/
│   └── app.py              # Streamlit web application
│
├── main.py                 # Entry point (reserved)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Key Components & Roles

### 🔹 1. Dataset (`data/`)

* Contains CT scan images
* Organized into:

  * `cancer/`
  * `normal/`

---

### 🔹 2. Notebook (`experiments.ipynb`)

* Core development file
* Performs:

  * Data loading
  * Preprocessing
  * Model training (KNN, SVM, CNN)
  * Evaluation
* Saves trained model to:

  ```
  models/cnn_model.h5
  ```

---

### 🔹 3. Model (`models/`)

* Stores trained CNN model
* Used during deployment
* Avoids retraining every time

---

### 🔹 4. Source Code (`src/`)

* Modular implementation
* `models.py` → defines CNN architecture
* `preprocessing.py` → handles image processing
* `utils.py` → helper utilities

---

### 🔹 5. Web App (`app/app.py`)

* Built using Streamlit
* Allows:

  * Upload CT scan image
  * Preprocess input
  * Predict cancer/normal
  * Display results

---

## 🤖 Model Training

### Models Used:

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Convolutional Neural Network (CNN)

### Why CNN?

CNN automatically extracts spatial features from images, making it more suitable for medical image analysis.

---

## 📊 Model Evaluation

Performance is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

CNN achieves the best performance due to its ability to learn complex patterns.

---

## 🚀 Running the Project

### 1. Train Model (if needed)

Open notebook:

```
notebooks/experiments.ipynb
```

Run all cells.

---

### 2. Run Application

```
streamlit run app/app.py
```

---

## ⚠️ Important Notes

* The model is trained only on CT scan images
* Predictions on unrelated images (e.g., selfies) are not reliable
* Input validation is implemented to restrict non-medical images

---

## 🧠 Conclusion

This project demonstrates:

* End-to-end ML + DL pipeline
* Image preprocessing and feature extraction
* Model comparison and evaluation
* Deployment using Streamlit

It highlights how deep learning (CNN) improves performance in medical image classification tasks.

---

## 🎯 Future Improvements

* Use larger datasets for better accuracy
* Add Grad-CAM for visualization
* Improve input validation using a dedicated classifier
* Deploy on cloud platform

---

## 👨‍💻 Author

Neha sanjay mane
MSc Computer Science (AI/ML)

---
