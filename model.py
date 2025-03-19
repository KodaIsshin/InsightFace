import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import insightface
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import random


df = pd.read_csv("modified.csv")
base_dir = os.path.dirname(os.path.abspath(__file__))
df["NewPath"] = df["NewPath"].apply(lambda x: os.path.join(base_dir, x))

#initialize face analysis model using InsightFace

model =  FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))

#InsightFace also uses embeddings, so I'll be following shaun with this function
def get_embeddings(image_path):
    embeddings = []
    no_face_detected = []
    valid_indices = []
    count = 0
    total_images = len(image_path)

    for idx, path in enumerate(image_path):
        count += 1
        print(f"Getting embedding {count}")
        try:
            #loading image and converting to array of RGB
            img = Image.open(path)
            img=  ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img_array = np.asarray(img)

            faces = model.get(img_array)

            if len(faces) == 0:
                print(f"Skipping {path} (No face detected)")
                no_face_detected.append(path)
                continue
            
            #grabbing embedding from the faces variable
            embedding = faces[0].embedding #embedding for first face (this is for selfies or pictures with only one face)

            #compare with shaun's values
            print(f"Sample embedding values: {embedding[:5]}")
            #sample values
            # if count % 50 == 0:
            #     print(f"Sample embedding values: {embedding[:5]}")
            
            embeddings.append(embedding)
            valid_indices.append(idx)

            print(f"Processed {count}/{total_images}")

            
        except Exception as e:
            print(f"Error for {path} : {e}")
            no_face_detected.append(path)

    return embeddings, no_face_detected, valid_indices

X, no_face_images, valid_indices = get_embeddings(df["NewPath"].tolist())
y = df["Name"].values  # Labels
df["Embedding"] = None

for idx, modidx in enumerate(valid_indices):
    df.at[modidx, 'Embedding'] = X[idx]

print("DONE")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train = np.array(X_train)

# Get original names for better reporting
y_train_names = le.inverse_transform(y_train)
y_test_names = le.inverse_transform(y_test)

# 1. SVM Classification
print("===== SVM Classifier =====")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Test SVM
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# Show some example predictions
print("\nSVM Example Predictions:")
for i in range(5):
    true_name = le.inverse_transform([y_test[i]])[0]
    pred_name = le.inverse_transform([y_pred_svm[i]])[0]
    proba = svm_model.predict_proba([X_test[i]])[0]
    confidence = proba[y_pred_svm[i]]
    print(f"True: {true_name:10} | Predicted: {pred_name:10} | Confidence: {confidence:.4f}")  # Ensure X contains your embeddings


# 2. Distance-based verification
print("\n===== Distance-based Verification =====")

# Create reference embeddings per person
reference_embeddings = {}
for name in le.classes_:
    name_idx = np.where(y_train_names == name)[0]
    reference_embeddings[name] = X_train[np.array(name_idx, dtype=int)]


# Test distance-based approach
correct_count = 0
print("Distance-based Example Predictions:")
for i in range(5):
    embedding = X_test[i]
    true_name = le.inverse_transform([y_test[i]])[0]
    
    # Calculate similarity to each class
    similarities = {}
    for name, refs in reference_embeddings.items():
        # Calculate cosine similarities
        sims = cosine_similarity([embedding], refs)[0]
        similarities[name] = np.mean(sims)
    
    # Get predicted name (highest similarity)
    pred_name = max(similarities, key=similarities.get)
    confidence = similarities[pred_name]
    true_sim = similarities[true_name]
    
    if pred_name == true_name:
        correct_count += 1
        
    print(f"True: {true_name:10} | Predicted: {pred_name:10} | Similarity: {confidence:.4f} | True similarity: {true_sim:.4f}")

# Calculate overall distance-based accuracy
y_pred_dist = []
for i in range(len(X_test)):
    embedding = X_test[i]
    similarities = {name: np.mean(cosine_similarity([embedding], refs)[0]) 
                   for name, refs in reference_embeddings.items()}
    pred_name = max(similarities, key=similarities.get)
    y_pred_dist.append(le.transform([pred_name])[0])

accuracy_dist = accuracy_score(y_test, y_pred_dist)
print(f"\nDistance-based Accuracy: {accuracy_dist:.4f}")
print(classification_report(y_test, y_pred_dist, target_names=le.classes_))


# 3. K-Nearest Neighbors (another good approach for face recognition)
print("\n===== KNN Classifier =====")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.4f}")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))

# 5. Create a confusion matrix to see which identities get confused
y_pred_names = le.inverse_transform(y_pred_dist)
conf_matrix = np.zeros((len(le.classes_), len(le.classes_)))

for i in range(len(y_test_names)):
    true_idx = np.where(le.classes_ == y_test_names[i])[0][0]
    pred_idx = np.where(le.classes_ == y_pred_names[i])[0][0]
    conf_matrix[true_idx, pred_idx] += 1

# Normalize confusion matrix
row_sums = conf_matrix.sum(axis=1)
conf_matrix_norm = conf_matrix / row_sums[:, np.newaxis]

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Face Recognition Confusion Matrix')
plt.tight_layout()
plt.show()
plt.savefig('confusion_matrix.png')
plt.close()


# Apply t-SNE to reduce embeddings to 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_train_2D = tsne.fit_transform(X_train)

# Create a scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train_2D[:, 0], y=X_train_2D[:, 1], hue=y_train_names, palette="tab10", s=50, alpha=0.7)

# Label the points with names
for i, name in enumerate(y_train_names):
    plt.text(X_train_2D[i, 0], X_train_2D[i, 1], name, fontsize=8, alpha=0.6)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Face Embeddings")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
plt.grid(True, alpha=0.3)
plt.show()






