# 🌾 Crop Recommendation System

This repository contains a **machine learning-based Crop Recommendation System**, which suggests the most suitable crop to grow based on soil and environmental conditions. The model is trained using a dataset that includes soil nutrients (NPK), temperature, humidity, pH level, and rainfall. The goal is to help farmers make data-driven decisions to optimize yield and sustainability.

## 🚀 Project Overview

This project explores how to use supervised machine learning techniques to classify and recommend the best crop to cultivate in a particular region based on input features such as:

- **Nitrogen (N)**
- **Phosphorus (P)**
- **Potassium (K)**
- **Temperature**
- **Humidity**
- **pH**
- **Rainfall**

The classification model was trained on an agricultural dataset and evaluated for accuracy and reliability using various metrics.

---

## 🛠️ Technologies Used

- Python 3.10+
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 📁 Dataset

The dataset used contains various environmental and soil parameters mapped to suitable crops. Each row represents one observation of field condition and its corresponding crop label.

| N | P | K | Temperature | Humidity | pH | Rainfall | Crop |
|---|---|---|-------------|----------|----|----------|------|
| 90 | 42 | 43 | 20.8°C | 82.0% | 6.5 | 202 mm | rice |

---

## 🔍 Exploratory Data Analysis

- Visualizations such as pairplots and correlation heatmaps were used to understand the relationship between features.
- Distribution plots helped identify patterns among different features across crops.
- Feature importance was derived using decision trees and random forest.

---

## 🧠 Model Training

Several classification models were trained and evaluated, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes

### ✅ Final Model Performance:
The **Random Forest Classifier** yielded the highest accuracy, with strong precision and recall scores. It was chosen for deployment due to its balance between interpretability and performance.

---

## 📦 Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/vivekkumarq/CropRecommend.git
   cd CropRecommend
