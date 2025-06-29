from flask import Flask, request, jsonify
import torch
from torchvision import models
import torch.nn as nn
from utils.transforms import preprocess_image
import os

# model
MODEL_PATH = "models/baseline_resnet18.pth"

# model setup
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10) 
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

class_names = [
    'battery', 'cardboard', 'clothes', 'food', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

app = Flask(__name__)

# main api route 

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = "temp.jpg"
    file.save(temp_path)

    # preprocessing
    image_tensor = preprocess_image(temp_path)

    # prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]

    os.remove(temp_path)

    return jsonify({"class": predicted_class})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)



