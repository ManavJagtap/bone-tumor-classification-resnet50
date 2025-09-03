import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# DEVICE SETUP (CPU)
# =========================
device = torch.device("cpu")

# =========================
# CLIP MODEL SETUP
# =========================
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

transform= clip_preprocess
# =========================
# TRANSFORM SETUP (518 x 518)
# Use CLIP mean/std but keep your requested size.
# =========================
#CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
#CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

#transform = transforms.Compose([
#    transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
#])

# =========================
# CUSTOM DATASET (unchanged)
# =========================
class ImageFolderDataset(Dataset):
    def __init__(self, folder_label_pairs, transform, limit_per_class=None):
        self.data = []
        self.labels = []
        self.transform = transform
        for folder, label in folder_label_pairs:
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            files.sort()
            if limit_per_class:
                files = files[:limit_per_class]
            for fname in files:
                self.data.append(os.path.join(folder, fname))
                self.labels.append(label)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]

# =========================
# FEATURE EXTRACTION (CLIP encode_image + L2 norm)
# =========================
@torch.no_grad()
def extract_features(data_loader):
    all_features, all_labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader, desc="🔍 Extracting CLIP features"):
        batch_imgs = batch_imgs.to(device)
        feats = clip_model.encode_image(batch_imgs)               # (B, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)          # L2 normalize
        all_features.append(feats.cpu().numpy().astype(np.float32))

        if isinstance(batch_labels, torch.Tensor):
            all_labels.extend(batch_labels.numpy().tolist())
        else:
            all_labels.extend(batch_labels)

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.array(all_labels)
    return X, y

# =========================
# PATHS TO DATASETS (same)
# =========================
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

# =========================
# LIMIT / LOADERS (same)
# =========================
LIMIT = None
print("📁 Loading training set...")
train_loader = DataLoader(ImageFolderDataset(train_pairs, transform, limit_per_class=LIMIT), batch_size=128, shuffle=False)

print("📁 Loading validation set...")
val_loader = DataLoader(ImageFolderDataset(val_pairs, transform, limit_per_class=LIMIT), batch_size=128, shuffle=False)

print("📁 Loading test set...")
test_loader = DataLoader(ImageFolderDataset(test_pairs, transform, limit_per_class=LIMIT), batch_size=128, shuffle=False)

# =========================
# EXTRACT FEATURES
# =========================
X_train, y_train = extract_features(train_loader)
X_val,   y_val   = extract_features(val_loader)
X_test,  y_test  = extract_features(test_loader)

# =========================
# COMBINE TRAIN + VAL FOR FINAL TRAINING
# =========================
X_combined = np.vstack((X_train, X_val)).astype(np.float32)
y_combined = np.hstack((y_train, y_val))
print("✅ Feature shape for training:", X_combined.shape)

# =========================
# TRAIN MODELS
# =========================
print("\n📈 Training Logistic Regression...")
logreg = LogisticRegression(max_iter=2000, solver="lbfgs")
logreg.fit(X_combined, y_combined)

print("\n📈 Training Linear SVM...")
linsvm = LinearSVC(C=1.0)
linsvm.fit(X_combined, y_combined)

# =========================
# EVALUATE: LOGISTIC REGRESSION
# =========================
print("\n🔎 Logistic Regression — Test Metrics")
y_pred_lr = logreg.predict(X_test)
print(classification_report(y_test, y_pred_lr, target_names=["Normal", "Cancer"]))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Cancer"], yticklabels=["Normal","Cancer"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix — Logistic Regression")
plt.tight_layout(); plt.savefig("clip_logreg_confusion_matrix.png"); plt.show(); plt.close()

y_prob_lr = logreg.predict_proba(X_test)[:, 1]
RocCurveDisplay.from_predictions(y_test, y_prob_lr, name="LogReg (CLIP)")
plt.plot([0,1],[0,1],"k--"); plt.title("ROC Curve — Logistic Regression")
plt.tight_layout(); plt.savefig("clip_logreg_roc_curve.png"); plt.show(); plt.close()

auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"🏅 LogReg ROC-AUC: {auc_lr:.4f}")

# =========================
# EVALUATE: LINEAR SVM
# =========================
print("\n🔎 Linear SVM — Test Metrics")
y_pred_svm = linsvm.predict(X_test)
print(classification_report(y_test, y_pred_svm, target_names=["Normal", "Cancer"]))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5,4))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Cancer"], yticklabels=["Normal","Cancer"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix — Linear SVM")
plt.tight_layout(); plt.savefig("clip_linsvm_confusion_matrix.png"); plt.show(); plt.close()

# LinearSVC doesn't output probabilities; use decision_function for ROC/AUC
y_score_svm = linsvm.decision_function(X_test)
RocCurveDisplay.from_predictions(y_test, y_score_svm, name="LinSVM (CLIP)")
plt.plot([0,1],[0,1],"k--"); plt.title("ROC Curve — Linear SVM")
plt.tight_layout(); plt.savefig("clip_linsvm_roc_curve.png"); plt.show(); plt.close()

auc_svm = roc_auc_score(y_test, y_score_svm)
print(f"🏅 Linear SVM ROC-AUC: {auc_svm:.4f}")

# =========================
# (Optional) Combined ROC plot for comparison
# =========================
plt.figure()
RocCurveDisplay.from_predictions(y_test, y_prob_lr, name="LogReg (CLIP)")
RocCurveDisplay.from_predictions(y_test, y_score_svm, name="LinSVM (CLIP)")
plt.plot([0,1],[0,1],"k--")
plt.title("ROC Curve — LogReg vs Linear SVM (CLIP)")
plt.tight_layout(); plt.savefig("clip_logreg_vs_linsvm_roc.png"); plt.show(); plt.close()