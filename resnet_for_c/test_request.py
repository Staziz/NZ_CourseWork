import requests
import os

# path
image_folder = 'test_images'
url = 'http://127.0.0.1:5000/predict'

# go through all images
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    print(f"\nImage: {image_name}")
    print(f"Status Code: {response.status_code}")
    print("Prediction:", response.json())



