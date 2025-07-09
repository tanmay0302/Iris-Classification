# 🌸 Iris Flower Classification using Machine Learning

This project builds a machine learning model to classify iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using features such as sepal length/width and petal length/width.

---

## 📁 Project Structure

├── Iris.csv # Dataset
├── knn_model.pkl # Trained KNN model (saved with joblib)
├── app.py # Streamlit web app for predictions
├── Iris_Classification.ipynb # Jupyter Notebook with step-by-step analysis
├── README.md # You're here

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris)
- **Features**:
  - SepalLengthCm
  - SepalWidthCm
  - PetalLengthCm
  - PetalWidthCm
- **Label**: Species (Setosa, Versicolor, Virginica)

---

## 🚀 How to Run the Project

### 🔧 1. Clone the Repository

```
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
```
📦 2. Install Dependencies
```
pip install -r requirements.txt

Or manually:
pip install pandas numpy seaborn matplotlib scikit-learn streamlit joblib
```
🌐 4. Run the Web App
```
app.py
```
🧠 Model Details
```
Algorithm: K-Nearest Neighbors (KNN)

Hyperparameter Tuning: GridSearchCV

Scaler: StandardScaler

Accuracy: ~96% on test data
```
📦 Requirements
```
Python 3.7+
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
streamlit
pandas
numpy
scikit-learn
joblib
```
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("knn_model.pkl")

st.title("🌸 Iris Flower Prediction App")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, step=0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, step=0.1)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, step=0.1)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"🌼 Predicted Iris Species: **{prediction[0]}**")
```
```
🚀 Live Demo

🔗 Try the Iris Classifier App Here:
https://iris-classification-a2tkewvsost4p8jxkutuxc.streamlit.app

This app allows you to:

📤 Upload your own Iris dataset

🔄 Retrain the model instantly using K-Nearest Neighbors

🌼 Predict the Iris species in real time with an intuitive interface

🧠 Skills Demonstrated
```
End-to-End Machine Learning Pipeline

Exploratory Data Analysis (EDA)

Feature Engineering & Scaling

Model Training & Tuning with GridSearchCV

Model Evaluation (Confusion Matrix, Accuracy, Precision, Recall)

Interactive Web App Deployment using Streamlit

GitHub Project Structuring & Documentation

