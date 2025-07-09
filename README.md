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
streamlit run app.py
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

