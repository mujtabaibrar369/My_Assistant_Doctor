from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# ============================
# Load TensorFlow Lite Models
# ============================

# Load Psychological Assessment Model
psych_interpreter = tf.lite.Interpreter(model_path='models/psychological_assessment_model.tflite')
psych_interpreter.allocate_tensors()
psych_input_index = psych_interpreter.get_input_details()[0]['index']
psych_output_index = psych_interpreter.get_output_details()[0]['index']

# Load General Health Model
health_interpreter = tf.lite.Interpreter(model_path='models/general_health_model.tflite')
health_interpreter.allocate_tensors()
health_input_index = health_interpreter.get_input_details()[0]['index']
health_output_index = health_interpreter.get_output_details()[0]['index']

# Load scalers and encoders
psych_scaler = joblib.load('models/psych_scaler.pkl')
psych_encoders = joblib.load('models/psych_encoders.pkl')
psych_label_encoder_y = joblib.load('models/psych_label_encoder_y.pkl')

health_scaler = joblib.load('models/health_scaler.pkl')
health_encoders = joblib.load('models/health_encoders.pkl')
health_label_encoder_y = joblib.load('models/health_label_encoder_y.pkl')

# ============================
# Load PyTorch Model for Skin Disease Classification
# ============================

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/skin_disease_model.pth'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define class labels
class_labels = [
    'Melanoma', 'Basal Cell Carcinoma', 'Melanocytic Nevi', 'Bengin Keratosis',
    'Seborrheic Keratoses and other Benign Tumors', 'Tinea', 'Warts'
]

# Define model architecture
def create_model(num_classes):
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skin_model = create_model(len(class_labels))
skin_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
skin_model.to(device)
skin_model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================
# Flask API Routes
# ============================

@app.route('/predict_psychological', methods=['POST'])
def predict_psychological():
    try:
        data = request.json
        required_features = [
            "Mood", "Anxious Social Scale", "Anxiety Triggers", "Sleep Quality",
            "Appetite Change", "Lack of Interest", "Enjoyable Activities",
            "Physical Anxiety Symptoms", "Concentration Difficulty", "Coping Strategies"
        ]
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing required features"}), 400

        # Prepare input data
        input_data = np.array([data[feature] for feature in required_features]).reshape(1, -1)
        for i, encoder in enumerate(psych_encoders):
            input_data[:, i] = encoder.transform(input_data[:, i])

        input_data = psych_scaler.transform(input_data).astype(np.float32)

        # Make prediction
        psych_interpreter.set_tensor(psych_input_index, input_data)
        psych_interpreter.invoke()
        predictions = psych_interpreter.get_tensor(psych_output_index)
        predicted_class = np.argmax(predictions)
        predicted_summary = psych_label_encoder_y.inverse_transform([predicted_class])[0]

        return jsonify({"Summary": predicted_summary})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_health', methods=['POST'])
def predict_health():
    try:
        data = request.json
        required_features = [
            "General Health", "Physical Symptoms", "Sleep Hours", "Physical Activity",
            "Weight/Appetite Changes", "Diet Quality", "Headaches/Dizziness/Nausea",
            "Stress Levels", "Substance Use", "Chronic Conditions"
        ]
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing required features"}), 400

        # Prepare input data
        input_data = np.array([data[feature] for feature in required_features]).reshape(1, -1)
        for i, encoder in enumerate(health_encoders):
            input_data[:, i] = encoder.transform(input_data[:, i])

        input_data = health_scaler.transform(input_data).astype(np.float32)

        # Make prediction
        health_interpreter.set_tensor(health_input_index, input_data)
        health_interpreter.invoke()
        predictions = health_interpreter.get_tensor(health_output_index)
        predicted_class = np.argmax(predictions)
        predicted_summary = health_label_encoder_y.inverse_transform([predicted_class])[0]

        return jsonify({"Summary": predicted_summary})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_skin_disease', methods=['POST'])
def predict_skin_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            # Preprocess image
            image = Image.open(file_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = skin_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()

            return jsonify({'prediction': class_labels[predicted_class]})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({'error': 'Invalid file type'}), 400

# ============================
# Run the Flask App
# ============================
if __name__ == '__main__':
    app.run(debug=True)
