import requests
import os

# path
image_folder = 'demonstration'
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

    try:
        result = response.json()
    except Exception as e:
        print("❌ Error: unable to parse response as JSON.")
        continue

    # проверка: был ли возврат ошибки
    if response.status_code == 200:
        print(f"✅ Prediction: {result['class']} (confidence: {result['confidence']})")
    else:
        print(f"⚠️ Error: {result.get('error', 'Unknown error')} (confidence: {result.get('max_prob', '?')})")



