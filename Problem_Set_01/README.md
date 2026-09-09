# Problem Set 02: Pneumonia Detection from Chest X-Ray Images

## 1. Project Overview
This project focuses on automated pneumonia detection from chest X-ray radiography using deep learning and convolutional neural networks (CNNs). The goal is to accurately distinguish between normal lungs and lungs exhibiting bacterial or viral pneumonia.

---

## 2. Dataset & Augmentation Pipeline
- **Dataset**: Chest X-ray images labeled as `Normal` and `Pneumonia`.
- **Image Preprocessing**:
  - Resized input images to uniform dimensions (e.g., $224 \times 224$).
  - Normalized pixel intensities to $[0, 1]$ or standardized to channel-wise mean and variance.
- **Data Augmentation**: Applied rotation, horizontal flipping, zoom, and shear transformations during training to expand dataset diversity and prevent overfitting on small training samples.

---

## 3. Methodology & Model Architecture
- **Architecture**: Deep Convolutional Neural Network (CNN) / Pretrained Transfer Learning backbone (e.g., ResNet / VGG / Custom ConvNet).
- **Feature Extractor**: Alternating convolutional layers (with ReLU activation) and max-pooling operations to extract spatial hierarchical patterns.
- **Classification Head**: Global Average Pooling followed by dense fully connected layers with Dropout.
- **Output Layer**: Sigmoid / Softmax activation for binary classification (`Normal` vs. `Pneumonia`).
- **Optimization**: Adam / SGD optimizer utilizing Binary Cross-Entropy loss.

---

## 4. Evaluation Metrics
- **Accuracy**: Overall classification accuracy.
- **Recall (Sensitivity)**: Critical metric to minimize false negatives in medical diagnostics.
- **Precision & F1-Score**: Evaluated to maintain balance between true detections and false alarms.
- **Confusion Matrix & Loss Curves**: Visualizing training vs. validation loss progression across epochs.

---

## 5. Summary of Results
- The model successfully learns distinctive radiographic features of pulmonary opacities indicative of pneumonia.
- Incorporating regularization and data augmentation helped mitigate overfitting and ensured consistent validation performance.
