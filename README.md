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


2.Install the dependencies:
pip install -r requirements.txt

Or manually install the required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn


3.Launch Jupyter Notebook:
jupyter notebook



🧪 Usage:
You can input custom values for:
Nitrogen, Phosphorus, Potassium
Temperature, Humidity
pH, Rainfall
The trained model will then predict the most suitable crop for the given conditions.
Example usage (from a script or interface):

📈 Results & Insights:
The model achieved high accuracy with the Random Forest Classifier.
Soil pH and rainfall were found to be highly influential in the crop selection.
The system can generalize well for unseen field conditions.


🔮 Future Enhancements:
🌍 Geo-location Integration: Automatically detect conditions based on coordinates.
☁️ Weather API Integration: Real-time weather fetching for dynamic predictions.
📱 Mobile/Web App Interface: Deploy as a Flask or Streamlit app for farmer accessibility.
🧪 Deep Learning Models: Test with ANN or RNN for more complex pattern recognition.
🗃️ Bigger Dataset: Train with a more diverse agricultural dataset across different regions and seasons
