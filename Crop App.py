from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import joblib # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore

app = Flask(__name__)
CORS(app) # للسماح بالاتصال من صفحة الويب

# تحميل الموديل والـ Scaler والـ LabelEncoder
try:
    model = joblib.load('best_knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder_target = joblib.load('label_encoder.pkl') # الخاص بعمود 'label'
except FileNotFoundError:
    print("Error: One or more .pkl files not found. Make sure to run the Jupyter notebook to save them.")
    exit()
except Exception as e:
    print(f"Error loading .pkl files: {e}")
    exit()

# أسماء الأعمدة بنفس الترتيب اللي اتدرب عليه الموديل
# من الـ notebook بتاعك، ترتيب الأعمدة في info قبل الـ drop للـ label هو:
# N, P, K, temperature, humidity, ph, rainfall
# هذا هو الترتيب الذي يجب أن يتوقعه الـ scaler والموديل
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # استقبال البيانات من الـ JSON request وتحويلها إلى float
        # الأسماء هنا يجب أن تطابق ما يرسله الـ JavaScript في Crop.html
        input_values = []
        # الترتيب هنا مهم لإنشاء الـ DataFrame بشكل صحيح لـ scaler.transform
        input_values.append(float(data['nitrogen']))
        input_values.append(float(data['phosphorus']))
        input_values.append(float(data['potassium']))
        input_values.append(float(data['temperature']))
        input_values.append(float(data['humidity'])) # HTML: humidity قبل ph_value
        input_values.append(float(data['ph_value'])) # HTML: ph_value هو الـ ph
        input_values.append(float(data['rainfall']))
        
        # إنشاء DataFrame بالترتيب الصحيح للـ features
        input_features_df = pd.DataFrame([input_values], columns=feature_columns)

        # عمل Scaling للمدخلات
        scaled_features = scaler.transform(input_features_df)

        # عمل التوقع
        prediction_encoded = model.predict(scaled_features)
        
        # تحويل التوقع من رقم إلى اسم المحصول
        prediction_label = label_encoder_target.inverse_transform(prediction_encoded)

        return jsonify({'prediction': prediction_label[0]})

    except KeyError as e:
        return jsonify({'error': f'Missing key in input data: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input data type: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)