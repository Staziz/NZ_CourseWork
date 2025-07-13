import os
import json
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- конфигурация ---
csv_path = 'clean_image_split.csv'            # очищенный CSV
dataset_root = Path('.')                      # папка, где лежит garbage-dataset
api_url = 'http://127.0.0.1:5000/predict'     # локальный API

# --- загрузка тестовых путей ---
df = pd.read_csv(csv_path)
df_test = df[df['split'] == 'test']
print(f'📄 всего тестовых изображений: {len(df_test)}')

# получаем список путей к изображениям
image_paths = [dataset_root / row['clean_path'] for _, row in df_test.iterrows()]

# --- инициализация ---
y_true = []
y_pred = []
confidences = []

# --- предсказания через API ---
for image_path in tqdm(image_paths):
    true_class = image_path.parts[-2]  # извлекаем класс из пути

    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': (image_path.name, img_file, 'image/jpeg')}
            response = requests.post(api_url, files=files)
            result = response.json()

            if 'error' in result:
                pred_class = 'unknown'
                confidence = 0.0
            else:
                pred_class = result['class']
                confidence = result['confidence']

            y_true.append(true_class)
            y_pred.append(pred_class)
            confidences.append(confidence)

    except Exception as e:
        print(f'❌ ошибка при обработке {image_path.name}: {e}')
        y_true.append(true_class)
        y_pred.append('unknown')
        confidences.append(0.0)

# --- сохраняем результаты ---
with open('eval_results.json', 'w') as f:
    json.dump({
        'y_true': y_true,
        'y_pred': y_pred,
        'confidences': confidences
    }, f, indent=2)

# --- отчёты и визуализация ---
filtered_true = [yt for yt, yp in zip(y_true, y_pred) if yp != 'unknown']
filtered_pred = [yp for yp in y_pred if yp != 'unknown']

print('\n📊 classification report (без unknown):')
print(classification_report(filtered_true, filtered_pred))

print('\n🧮 confusion matrix:')
cm = confusion_matrix(filtered_true, filtered_pred, labels=sorted(set(filtered_true)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(filtered_true)))

# визуализация и сохранение
plt.figure(figsize=(12, 10))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix (без unknown)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print('📁 confusion_matrix.png сохранена')