## Bone Tumor Classification (ResNet50, DINOv2, CLIP)

Deep learning pipelines for binary classification of MRI bone scans into Cancerous and Normal categories.
Models included:

- ResNet50 (CNN baseline) with Grad-CAM interpretability
- DINOv2 (Vision Transformer) embeddings + Logistic Regression, Linear SVM, XGBoost classifiers
- CLIP (ViT-B/32) embeddings + Logistic Regression, Linear SVM classifiers

## Why This Matters

- Diagnosing bone tumors from MRI scans is time-consuming and subjective.
- Delays or errors can affect treatment outcomes.
- This project aims to build a **fast, interpretable AI tool** to assist in diagnosis.

## Model Architecture

- ResNet50 (CNN Baseline)

Pretrained on ImageNet, fine-tuned for MRI data

Layers: Global Average Pooling → Dense (ReLU) → Dropout → Sigmoid output

Task: Binary classification (Cancer vs. Normal)

Explainability: Grad-CAM heatmaps

- DINOv2 (Vision Transformer)

Self-supervised ViT-Base/32 model

Extracts 768-d embeddings from MRI scans

Classifiers applied: Logistic Regression, Linear SVM, XGBoost

- CLIP (Contrastive Language–Image Pretraining)

Uses ViT-B/32 image encoder

Extracts 512-d embeddings for classification

Classifiers applied: Logistic Regression & Linear SVM

## Project Workflow

1. Data Preprocessing 
   Resize to 224×224, normalize, and apply augmentations (flip, zoom, rotate).
   Normalize (ImageNet mean/std or [0,1] scaling)
   Apply augmentations (flip, zoom, rotate)
   
2. Model Training/ Feature Extraction
   ResNet50 trained end-to-end
   DINOv2 & CLIP used as frozen feature extractors + external classifiers
    
3. Performance Evaluation
   Metrics: Accuracy, Sensitivity, Specificity, ROC-AUC
   Confusion matrices & ROC curves generated for all classifiers
     
4. Visualization (Grad-CAM)
   Grad-CAM applied to ResNet50 predictions to localize tumor-influenced regions
   
## Files Included

- test3resnet50.py: ResNet50 training & Grad-CAM pipeline
- DINOv2_Updated.py: Feature extraction & classification with DINOv2
- Clip.py: Feature extraction & classification with CLIP
- gradcam.py: Grad-CAM visualization functions
- README.md: Project overview & usage
  
## Results (Test Set)

| Model            | Accuracy | Sensitivity | Specificity | AUC   |
| ---------------- | -------- | ----------- | ----------- | ----- |
| ResNet50         | 91.7%    | 92%         | 92%         | 0.978 |
| DINOv2 + LogReg  | 97%      | 97%         | 98%         | 0.996 |
| DINOv2 + SVM     | 97%      | 97%         | 97%         | 0.995 |
| DINOv2 + XGBoost | 98%      | 97.1%       | 98.4%       | 0.999 |
| CLIP + LogReg    | 89%      | 86%         | 92%         | 0.970 |
| CLIP + SVM       | 92%      | 90%         | 93%         | 0.983 |

## Future Work

- Fine-tune deeper ResNet50 layers  
- Extend to multiclass tasks (e.g., tumor type or stage)  
- Experiment with other CNNs (DenseNet, EfficientNet)  
- Train on larger, more diverse MRI datasets
- Create a Medical image classification using a custom trained model.
- Fine-tune DINOv2 and CLIP on medical imaging datasets
- Extend to multi-class classification (different tumor subtypes)
- Incorporate multimodal pipelines (MRI + text reports)
- Evaluate performance on external datasets for robustness

## Academic Context

This project was created for the MSc Bioinformatics Dessertation at the University of Liverpool.

## License

For educational and academic use only.


