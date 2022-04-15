
from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here


@app.route('/', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier
        model3 = joblib.load("model3.pkl")

        # Get values through input bars
        Nitrogen = request.form.get("Nitrogen")
        Phosphorous = request.form.get("Phosphorous")
        Potassium = request.form.get("Potassium")
        temperature = request.form.get("temperature")
        Ph = request.form.get("Ph")
        humidity = request.form.get("humidity")
        rainfall = request.form.get("rainfall")

        # Put inputs to dataframe
        X = pd.DataFrame([[Nitrogen, Phosphorous, Potassium, temperature, Ph, humidity, rainfall]], columns=[
                         "N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

        # Get prediction
        prediction = model3.predict(X)[0]
        text = "The recommended Crop: "
    else:
        prediction = ""
        text = ""
    return render_template("index1.html", output=text+prediction)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
