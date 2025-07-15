# 🌸 Iris Flower Species Prediction Web App
A simple and interactive machine learning web application built with Flask that predicts the species of Iris flower based on user input for sepal and petal dimensions.
It uses a pre-trained model (iris_model.pkl) and provides a smooth user experience via a responsive HTML interface.

# 🔍 Features
-- 🚀 Built with Flask – lightweight and easy to deploy

-- 🧠 ML Model (Pickle) – trained on the Iris dataset

-- 📊 4 Input Features – Sepal Length, Sepal Width, Petal Length, Petal Width

-- 🌺 3 Iris Species Prediction

Iris-setosa

Iris-versicolor

Iris-virginica

-- 🎨 Beautiful UI – designed using pure HTML + CSS

# 📁 Folder Structure

iris-flask-app/
│
├── app.py                  # Flask backend
├── iris_model.pkl          # Trained ML model
└── templates/
    ├── index.html          # Input form
    └── result.html         # Output result
# 🖥️ Screenshots
🔹 Input Page (index.html)
Users input flower dimensions via an interactive form. https://github.com/Sudipta-Mitra/Iris-dataset/blob/main/Screenshot%20(47).png

🔹 Result Page (result.html)
Displays predicted Iris species using model output.

---
# 📦 Requirements
- Python 3.6+

- Flask

- scikit-learn

- numpy

You can install dependencies with:
- pip install flask scikit-learn numpy
---
# ▶️ Run the App
- Clone or download the repository

- Make sure iris_model.pkl is in the root folder

- Run:

python app.py
Visit http://127.0.0.1:5000 in your browser
---
# 🧠 About the Model
- Trained on the classic Iris dataset from sklearn.datasets

- Likely using a classification model such as Logistic Regression, KNN, or RandomForest

- Output is mapped from class labels (0, 1, 2) to species names
---
# 💡 Use Case
This project is perfect for:

Learning Flask web deployment

Showcasing ML integration in web development

Educational demos or mini-projects

