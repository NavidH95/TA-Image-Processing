import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Configuration
classes = ['headphone', 'soccer_ball', 'watch']  # Alphabetically sorted
training_data_size_percent = 20
number_of_clusters = 500
ratio_of_strong_features = 0.8
svm_kernel = 'rbf'
svm_c = 0.1

def load_images_from_folder(folder, label):
    images = []
    labels = []
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Warning: The folder {folder} does not exist. Skipping.")
        return images, labels
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Load and balance datasets
data, labels = [], []
for class_name in classes:
    folder_path = os.path.join('BoW/101_ObjectCategories', class_name)
    images, lbls = load_images_from_folder(folder_path, class_name)
    data.extend(images)
    labels.extend(lbls)

if not data:
    raise ValueError("No images found. Please check your dataset paths and ensure the directories exist.")

min_class_count = min([labels.count(c) for c in classes if labels.count(c) > 0])
balanced_data, balanced_labels = [], []
for class_name in classes:
    class_images = [data[i] for i in range(len(data)) if labels[i] == class_name]
    balanced_data.extend(class_images[:min_class_count])
    balanced_labels.extend([class_name] * min_class_count)

# Train-test split
train_data, val_data, train_labels, val_labels = train_test_split(
    balanced_data, balanced_labels, 
    test_size=(100 - training_data_size_percent) / 100, 
    stratify=balanced_labels
)

# Extract features using SURF
def extract_features(images):
    surf = cv2.SIFT_create()  # SIFT used as SURF might not be available
    descriptors = []
    for img in images:
        keypoints, des = surf.detectAndCompute(img, None)
        if des is not None:
            descriptors.extend(des.astype(np.float32))
    return descriptors

def cluster_features(descriptors, k):
    descriptors = np.array(descriptors, dtype=np.float32)
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(descriptors)
    return kmeans

def create_feature_histogram(images, kmeans):
    surf = cv2.SIFT_create()
    histograms = []
    for img in images:
        keypoints, des = surf.detectAndCompute(img, None)
        if des is not None:
            labels = kmeans.predict(des)
            histogram, _ = np.histogram(labels, bins=range(kmeans.n_clusters + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(kmeans.n_clusters))
    return np.array(histograms)

train_descriptors = extract_features(train_data)
kmeans = cluster_features(train_descriptors, number_of_clusters)

# Generate feature histograms for training and validation data
train_histograms = create_feature_histogram(train_data, kmeans)
val_histograms = create_feature_histogram(val_data, kmeans)

# Encode labels
label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(train_labels)
encoded_val_labels = label_encoder.transform(val_labels)

# Train SVM classifier
svm = SVC(kernel=svm_kernel, C=svm_c, probability=True)
svm.fit(train_histograms, encoded_train_labels)

# Evaluate classifier
def evaluate_model(model, histograms, labels):
    predictions = model.predict(histograms)
    conf_matrix = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    return conf_matrix, accuracy

print('---EVALUATING CLASSIFIER ON TRAINING SET---')
conf_matrix_train, train_accuracy = evaluate_model(svm, train_histograms, encoded_train_labels)
print('Confusion Matrix (Train):\n', conf_matrix_train)
print('Accuracy (Train):', train_accuracy)

print('---EVALUATING CLASSIFIER ON VALIDATION SET---')
conf_matrix_val, val_accuracy = evaluate_model(svm, val_histograms, encoded_val_labels)
print('Confusion Matrix (Validation):\n', conf_matrix_val)
print('Accuracy (Validation):', val_accuracy)

tran_val_avg_accuracy = (train_accuracy + val_accuracy) / 2
print(f'The training and validation average accuracy is {tran_val_avg_accuracy:.2f}.')

# Test on unseen images
def try_load_test_images(folder):
    images, labels = load_images_from_folder(folder, 'unknown')
    return images, labels

test_data, test_labels = try_load_test_images('BoW/Test_Data')
if test_data:
    test_histograms = create_feature_histogram(test_data, kmeans)
    encoded_test_labels = label_encoder.transform(test_labels)

    print('---EVALUATING CLASSIFIER ON TEST SET---')
    conf_matrix_test, test_accuracy = evaluate_model(svm, test_histograms, encoded_test_labels)
    print('Confusion Matrix (Test):\n', conf_matrix_test)
    print(f'The test accuracy is {test_accuracy:.2f}.')
else:
    print("No test data found. Skipping test evaluation.")
