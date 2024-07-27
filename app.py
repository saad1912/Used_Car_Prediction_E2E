from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train  pipeline 
def training():
    os.system("python main.py")
    return render_template("train_success.html")

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            brand = request.form['brand']
            model_name = request.form['model']
            vehicle_age = int(request.form['vehicle_age'])
            km_driven = int(request.form['km_driven'])
            seller_type = request.form['seller_type']
            fuel_type = request.form['fuel_type']
            transmission_type = request.form['transmission_type']
            mileage = float(request.form['mileage'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])
            seats = int(request.form['seats'])

            # Create DataFrame for prediction
            input_data = pd.DataFrame([[brand, model_name, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats]],
                                      columns=['brand', 'model', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power', 'seats'])

            obj = PredictionPipeline()
            prediction = obj.predict(input_data)

            return render_template('results.html', prediction=str(prediction[0]))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something is wrong'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
