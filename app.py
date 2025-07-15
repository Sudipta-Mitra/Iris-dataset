from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Map from numeric prediction to species name
label_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[x]) for x in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        prediction = model.predict([features])[0]

        species = label_map.get(prediction, "Unknown")

        return render_template("result.html", prediction_text=f"Predicted Iris Species: {species}")
    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
