import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from skimage.feature import hog

# Impor 'SVC' (Support Vector Classifier) dari 'sklearn.svm'
from sklearn.svm import SVC

# Impor 2 fungsi dari 'sklearn.model_selection'
# 1. Untuk LeaveOneOut
# 2. Untuk cross_val_predict
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# Impor 3 fungsi dari 'sklearn.metrics'
# 1. Untuk classification_report
# 2. Untuk confusion_matrix
# 3. Untuk accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# --- Konfigurasi ---
# Sesuaikan path ini
FILE_PATH = r"C:\Users\HP\assignment\4212301026_ATS VISIONMECHINE\archive\emnist-letters-train.csv"  

# Dibuat lebih kecil agar proses belajar/uji coba lebih cepat
SAMPLES_PER_CLASS = 500 
NUM_CLASSES = 26
TOTAL_SAMPLES = NUM_CLASSES * SAMPLES_PER_CLASS

# Parameter HOG Disesuaikan dengan hasil Grid Search
PPC = (4, 4)  
CPB = (2, 2)  

# Parameter SVM Disesuaikan dengan hasil Grid Search
SVM_KERNEL = 'linear'
SVM_C = 0.1

# Fungsi ini sudah lengkap dari kode aslinya
def fix_orientation(img):
    """Fixes the EMNIST image orientation from the CSV format."""
    img_fixed = np.fliplr(img.T)
    return img_fixed

print("--- Load Dataset ---")
try:
    data_frame = pd.read_csv(FILE_PATH, header=None)
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print(f"ERROR : Filepath or File not found {FILE_PATH}")
    exit()

# Lengkapi bagian ini untuk memisahkan label dan gambar
labels_full = None
images_flat = None
# tuliskan script anda disini
# HINT: Gunakan .iloc. Kolom 0 adalah label, sisanya adalah gambar.
labels_full = data_frame.iloc[:, 0].values
images_flat = data_frame.iloc[:, 1:].values.astype('uint8')
# akhir script

images_raw = images_flat.reshape(-1, 28, 28)
images_full = np.array([fix_orientation(img) for img in images_raw])
print("Image orientation fixed.\n")


print("--- Data Sampling ---")
sampled_images = []
sampled_labels = []

print(f"Taking {SAMPLES_PER_CLASS} samples per class...")
for i in range(1, NUM_CLASSES + 1):
    class_indices = np.where(labels_full == i)[0]
    
    # Lengkapi bagian ini untuk mengambil indeks acak
    random_indices = None
    # tuliskan script anda disini
    # HINT: Gunakan np.random.choice(...)
    random_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)
    # akhir script
    
    sampled_images.append(images_full[random_indices])
    sampled_labels.append(labels_full[random_indices])

X_data = np.concatenate(sampled_images, axis=0)
y_data = np.concatenate(sampled_labels, axis=0)
print(f"Sampling complete. Total data for this experiment : {X_data.shape[0]} samples.\n")


print("--- HOG Feature Extraction ---")
hog_features = []
start_time = time.time()

print("Processing images for HOG feature extraction...")
for image in X_data:
    # Lengkapi bagian ini untuk mengekstrak fitur HOG
    features = None
    # tuliskan script anda disini
    # HINT: Panggil fungsi hog() dengan parameter PPC dan CPB
    features = hog(image, pixels_per_cell=PPC, cells_per_block=CPB, visualize=False, feature_vector=True)
    # akhir script
    hog_features.append(features)

X_features = np.array(hog_features)
end_time = time.time()
print(f"HOG extraction finished in {end_time - start_time:.2f} seconds.")
print(f"HOG feature dataset shape : {X_features.shape}\n")


print("--- Model Evaluation with LOOCV ---")

# Lengkapi bagian ini untuk membuat model SVM
model = None
# tuliskan script anda disini
# HINT: Buat instance dari SVC()
model = SVC(kernel=SVM_KERNEL, C=SVM_C)
# akhir script
print(f"SVM model prepared with kernel ='{SVM_KERNEL}' and C ={SVM_C}")

# Lengkapi bagian ini untuk membuat object LeaveOneOut
loo = None
# tuliskan script anda disini
loo = LeaveOneOut()
# akhir script
print(f"Validation method: Leave-One-Out (will run {TOTAL_SAMPLES} iterations).\n")

print("===============================================================")
print("STARTING LOOCV EVALUATION.....)")
start_cv_time = time.time()

# Lengkapi bagian ini untuk menjalankan cross-validation
y_pred = None
# tuliskan script anda disini
# HINT: Gunakan cross_val_predict(). Masukkan model, data (X, y), dan cv=loo
y_pred = cross_val_predict(model, X_features, y_data, cv=loo, n_jobs=-1)
# akhir script

end_cv_time = time.time()
total_minutes = (end_cv_time - start_cv_time) / 60
print(f"\nLOOCV evaluation finished in {total_minutes:.2f} minutes.\n")

print("--- Display Performance Results ---")

# Lengkapi bagian ini untuk menghitung akurasi
accuracy = 0
# tuliskan script anda disini
# HINT: Gunakan accuracy_score() untuk membandingkan y_data dan y_pred
accuracy = accuracy_score(y_data, y_pred)
# akhir script
print(f"Accuracy: {accuracy * 100:.4f}%\n")

print("Classification Report (Precision, Recall, F1-Score):")
report_labels = list(range(1, NUM_CLASSES + 1))
target_names = [chr(ord('A') + i - 1) for i in report_labels]

# Lengkapi bagian ini untuk membuat classification report
report = ""
# tuliskan script anda disini
# HINT: Gunakan classification_report()
report = classification_report(y_data, y_pred, labels=report_labels, target_names=target_names, digits=4)
# akhir script
print(report)

print("Generating Confusion Matrix plot...")
# Lengkapi bagian ini untuk membuat confusion matrix
cm = None
# tuliskan script anda disini
# HINT: Gunakan confusion_matrix()
cm = confusion_matrix(y_data, y_pred, labels=report_labels)
# akhir script

plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix (LOOCV - HOG + SVM {SVM_KERNEL.capitalize()})', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

plt.savefig('confusion_matrix_loocv_belajar.png')
print("Confusion Matrix plot saved as 'confusion_matrix_loocv_belajar.png'")
plt.show()