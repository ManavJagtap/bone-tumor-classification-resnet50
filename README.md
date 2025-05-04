## bone-tumor-classification-resnet50
ResNet50-based deep learning pipeline for binary classification of bone tumor MRI images.

## Bone Tumor Detection Using ResNet50

This project uses a deep learning model (ResNet50) to automatically classify MRI bone scans as cancerous or non-cancerous. It also applies Grad-CAM to visualize which parts of the image influenced the model’s prediction — enhancing interpretability.

## Why This Matters

- Diagnosing bone tumors from MRI scans is time-consuming and subjective.
- Delays or errors can affect treatment outcomes.
- This project aims to build a **fast, interpretable AI tool** to assist in diagnosis.

## Model Architecture

- Base model: ResNet50 (pretrained on ImageNet)
- Added layers: Global Average Pooling → Dense (ReLU) → Dropout → Output (Sigmoid)
- Task: Binary classification — Cancer vs. Normal

## Project Workflow

1. Data Preprocessing 
   Resize to 224×224, normalize, and apply augmentations (flip, zoom, rotate).
   
2. Model Training
   Compile and train with early stopping and checkpointing.
   
3. Performance Evaluation
   Plot accuracy/loss curves, ROC-AUC, and confusion matrix.
   
4. Visualization (Grad-CAM)
   Generate heatmaps to localize tumor-influenced regions.

## Files Included

- train_model.py: Training script with full data pipeline  
- gradcam.py: Function to generate Grad-CAM overlays  
- README.md: Project overview and instructions

## Sample Outputs

- Accuracy: 93% on validation data  
- ROC AUC: High area under the ROC curve  
- Confusion Matrix: Balanced predictions  
- Grad-CAM: Localizes features contributing to decisions

## Future Work

- Fine-tune deeper ResNet50 layers  
- Extend to multiclass tasks (e.g., tumor type or stage)  
- Experiment with other CNNs (DenseNet, EfficientNet)  
- Train on larger, more diverse MRI datasets

## Academic Context

This project was created for the MSc Bioinformatics Dessertation at the University of Liverpool.

## License

For educational and academic use only.


