import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from PIL import Image
import mediapipe as mp
import cv2

# === Paths ===
data_path = Path('/Users/sillahbabar/Desktop/oulu/emotic/archive_emot/annots_arrs')
img_dir = Path('/Users/sillahbabar/Desktop/oulu/emotic/archive_emot/img_arrs/')
crop_dir = Path('/Users/sillahbabar/Desktop/oulu/emotic/archive_emot/img_arrs/')

csv_path = data_path / 'annot_arrs_extra_train.csv'
output_csv_path = data_path / 'annot_arrs_extra_train_with_visibility.csv'

# === Load CSV ===
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} samples")

# === Initialize MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# === Compute face visibility for each crop ===
face_visibility_scores = []

print("Processing crops for face visibility estimation...")

for idx, row in df.iterrows():
    crop_npy_path = crop_dir / row['Crop_name']
    try:
        if crop_npy_path.exists():
            # Load crop image (numpy array)
            crop_array = np.load(crop_npy_path)
            
            # Normalize if needed and convert to uint8 RGB
            if crop_array.max() > 1.0:
                crop_array = (crop_array / 255.0).astype('float32')
            crop_rgb = (crop_array * 255).astype('uint8')
            
            # Convert grayscale to RGB if necessary
            if crop_rgb.ndim == 2:
                crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2RGB)
            elif crop_rgb.shape[2] == 1:
                crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2RGB)

            # Run MediaPipe Face Mesh
            results = face_mesh.process(crop_rgb)
            print("results: ", results)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                print("landmarks: ",len(landmarks))
                print("landmark: ", landmarks[0])
                print("land mark visibility: ", [l for l in landmarks if l.visibility])

                visible_landmarks = [l for l in landmarks if 0 <= l.x <= 1 and 0 <= l.y <= 1]
                percent_visible = len(visible_landmarks) / 468 * 100
            else:
                percent_visible = 0.0  # no face detected
            
            face_visibility_scores.append(percent_visible)
        else:
            print(f"[{idx}] Crop file not found: {crop_npy_path}")
            face_visibility_scores.append(np.nan)
    
    except Exception as e:
        print(f"[{idx}] Error processing crop {crop_npy_path}: {e}")
        face_visibility_scores.append(np.nan)

# Close MediaPipe resources
face_mesh.close()

# === Add new column to DataFrame ===
df['Face_Visibility'] = face_visibility_scores

# === Save updated CSV ===
df.to_csv(output_csv_path, index=False)
print(f"Saved updated CSV with Face_Visibility column to:\n{output_csv_path}")

# === Plot histogram of face visibility ===
plt.figure(figsize=(8, 5))
sns.histplot(df['Face_Visibility'].dropna(), bins=30, color='skyblue', kde=False)
plt.xlabel('Face Visibility (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Face Visibility in Training Crops')
plt.grid(True)
plt.tight_layout()
plt.savefig('face_visibility_distribution.png', dpi=200)
plt.show()
