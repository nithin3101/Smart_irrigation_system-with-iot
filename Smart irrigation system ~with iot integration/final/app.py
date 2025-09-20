from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to access APIs

# Load model and encoders
model = joblib.load('water_required_regression_model.pkl')
encoders = joblib.load('regression_crop_encoder.pkl')

# Global storage for sensor data
sensor_data = {
    'soilMoisture': None,
    'temperature': None,
    'humidity': None
}

# For rendering live data
latest_data = {
    'temperature': None,
    'humidity': None,
    'soil': None
}

@app.route('/')
def index():
    # Serve the index.html page with live sensor data
    return render_template('index.html')

@app.route('/predict_irrigation')
def predict_irrigation():
    # Serve the predict_irrigation.html page with latest sensor values
    return render_template('predict_irrigation.html', data=latest_data)

@app.route('/sensor_input', methods=['POST'])
def receive_sensor_data():
    try:
        # Support both form data and JSON
        if request.is_json:
            data = request.get_json()
            soil = float(data.get('soilMoisture'))
            humidity = float(data.get('humidity'))
            temperature = float(data.get('temperature'))
        else:
            soil = float(request.form['soilMoisture'])
            humidity = float(request.form['humidity'])
            temperature = float(request.form['temperature'])

        # Update global storage
        sensor_data['soilMoisture'] = soil
        sensor_data['humidity'] = humidity
        sensor_data['temperature'] = temperature

        latest_data['soil'] = soil
        latest_data['humidity'] = humidity
        latest_data['temperature'] = temperature

        print("✅ Data received:", latest_data)
        return jsonify({'message': 'Sensor data received successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    return jsonify(sensor_data)

@app.route('/live_data', methods=['GET'])
def live_data():
    return jsonify(latest_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        crop = request.form['crop']
        acreage = float(request.form['acreage'])

        # Use the latest sensor data
        soil = sensor_data['soilMoisture']
        humidity = sensor_data['humidity']
        temp = sensor_data['temperature']

        if soil is None or humidity is None or temp is None:
            return jsonify({'error': 'Sensor data not available'}), 400

        suggestion = "Irrigate" if soil < 30 else "Do not irrigate"
        prediction = None

        if suggestion == "Irrigate":
            input_df = pd.DataFrame({
                'Crop': [crop],
                'Acreage': [acreage],
                'CropDays': [70],
                'Temperature': [temp],
                'SoilMoisture': [soil],
                'Humidity': [humidity],
                'Evapotranspiration': [5.2],
                'CropCoefficient': [1.05]
            })

            input_df['Crop'] = encoders['Crop'].transform(input_df['Crop'])
            prediction = round(model.predict(input_df)[0], 2)

        return render_template('predict_irrigation.html',
                               soilMoisture=soil,
                               temperature=temp,
                               humidity=humidity,
                               suggestion=suggestion,
                               prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
