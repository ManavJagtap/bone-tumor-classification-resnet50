import os
import torch
import timm
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# DEVICE SETUP
device = torch.device("cpu")

# DINOv2 MODEL
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
model.eval().to(device)

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# CUSTOM DATASET
class ImageFolderDataset(Dataset):
    def __init__(self, folder_label_pairs, transform, limit_per_class=None):
        self.data = []
        self.labels = []
        self.transform = transform
        for folder, label in folder_label_pairs:
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if limit_per_class:
                files = files[:limit_per_class]
            for fname in files:
                self.data.append(os.path.join(folder, fname))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

# FEATURE EXTRACTION
def extract_features(dataloader):
    all_features = []
    all_labels = []
    for batch_imgs, batch_labels in tqdm(dataloader, desc="🔍 Extracting features"):
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            feats = model.forward_features(batch_imgs)
            pooled_feats = feats.mean(dim=1)
        all_features.append(pooled_feats.cpu().numpy())
        all_labels.extend(batch_labels.numpy())
        gc.collect()
    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.array(all_labels)
    return X, y

# DATASET PATHS
train_pairs = [
    ("/home/hlmjagta/bone_tumor_data/bone cancer detection.v1i.multiclass/cancer train dataset/cancer", 1),
    ("/home/hlmjagta/bone_tumor_data/bone cancer detection.v1i.multiclass/cancer train dataset/normal", 0)
]
val_pairs = [
    ("/home/hlmjagta/bone_tumor_data/bone cancer detection.v1i.multiclass/cancer valid dataset/cancer", 1),
    ("/home/hlmjagta/bone_tumor_data/bone cancer detection.v1i.multiclass/cancer valid dataset/normal", 0)
]
test_pairs = [
    ("/home/hlmjagta/bone_tumor_data/bone cancer detection.v1i.multiclass/cancer test dataset/cancer", 1),
    ("/home/hlmjagta/bone_tumor_data/bone cancer detection.v1i.multiclass/cancer test dataset/normal", 0)
]

# LOAD DATA
LIMIT = None
train_loader = DataLoader(ImageFolderDataset(train_pairs, transform, LIMIT), batch_size=128, shuffle=False)
val_loader = DataLoader(ImageFolderDataset(val_pairs, transform, LIMIT), batch_size=128, shuffle=False)
test_loader = DataLoader(ImageFolderDataset(test_pairs, transform, LIMIT), batch_size=128, shuffle=False)

# EXTRACT FEATURES
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)
X_test, y_test = extract_features(test_loader)

# COMBINE TRAIN + VAL
X_combined = np.vstack((X_train, X_val))
y_combined = np.hstack((y_train, y_val))

# LOGISTIC REGRESSION
print("📈 Training Logistic Regression...")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_combined, y_combined)
y_pred_logreg = logreg.predict(X_test)
print("✅ Logistic Regression Report:")
print(classification_report(y_test, y_pred_logreg))

# LINEAR SVM
print("📈 Training Linear SVM...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_combined, y_combined)
y_pred_svm = svm.predict(X_test)
print("✅ Linear SVM Report:")
print(classification_report(y_test, y_pred_svm))

# --- Evaluation Plots ---
def plot_confusion_matrix(y_true, y_pred, Confusion_matrix, filename):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 {Confusion_matrix} Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Cancer"], yticklabels=["Normal", "Cancer"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(Confusion_matrix)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_scores, ROC_Curve, label, filename):
    RocCurveDisplay.from_predictionss(y_true, y_scores, name=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(ROC_Curve)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    auc = roc_auc_score(y_true, y_scores)
    print(f"🏅 {label} AUC: {auc:.4f}")

# LOGISTIC REGRESSION EVALUATION
plot_confusion_matrix(y_test, y_pred_logreg, "Logistic Regression", "logreg_confusion_matrix.png")
logreg_probs = logreg.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, logreg_probs, "Logistic Regression ROC Curve", "LogReg", "logreg_roc_curve.png")

# SVM EVALUATION
plot_confusion_matrix(y_test, y_pred_svm, "Linear SVM", "svm_confusion_matrix.png")
svm_probs = svm.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, svm_probs, "SVM ROC Curve", "SVM", "svm_roc_curve.png")
