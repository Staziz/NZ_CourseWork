from flask import Flask, request, jsonify
import torch
from torchvision import models
import torch.nn as nn
from utils.transforms import preprocess_image
import os

# model
MODEL_PATH = "models/baseline_resnet18_new.pth"

# model setup
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10) 
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

class_names = [
    'battery', 'biological','cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]
# threshhold
CONFIDENCE_THRESHOLD = 0.8

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
        # softmax - probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted_idx = torch.max(probs, dim=1)

        # low probability - error 
        if max_prob.item() < CONFIDENCE_THRESHOLD:
            os.remove(temp_path)
            return jsonify({"error": "low confidence", "max_prob": round(max_prob.item(), 3)}), 400

        # return class if probability > threshold
        predicted_class = class_names[predicted_idx.item()]

    os.remove(temp_path)

    return jsonify({"class": predicted_class, "confidence": round(max_prob.item(), 3)})

# activate server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5691)



