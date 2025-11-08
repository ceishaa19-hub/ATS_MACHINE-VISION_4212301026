CEISHA ALIYA KISNAWAN
4212301026
MEKATRONIKA 5A PAGI - ATS MACHINE VISION

## Create a program for handwritten character classification on the EMNIST (Extended MNIST) dataset using HOG Feature Extraction with a Support Vector Machine (SVM) classifier

## Introduction
This experiment demonstrates the implementation of a handwritten character recognition system using the EMNIST Letters dataset. The system employs the Histogram of Oriented Gradients (HOG) feature descriptor to extract image features and the Support Vector Machine (SVM) classifier for recognition.
A Leave-One-Out Cross Validation (LOOCV) strategy is applied to evaluate model performance thoroughly, ensuring every sample contributes once as a test case.

---

## Methodology
1. **Dataset Preparation**  
   - Dataset used: emnist-letters-train.csv
   - The dataset contains grayscale images of handwritten alphabet letters (A–Z).
   - Each image is 28×28 pixels and flattened in the CSV format.
   - A total of 13,000 samples (26 classes × 500 samples/class) were used for this experiment to reduce computational load.

2. **Preprocessing**  
   - Each image from the CSV file was reshaped and reoriented using a custom function:
     def fix_orientation(img):
     img_fixed = np.fliplr(img.T)
     return img_fixed
   - This correction ensures the images align correctly with EMNIST’s original orientation.

3. **Sampling**  
   - From each of the 26 letter classes (A–Z), 500 random samples were selected using:
     random_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)

4. **Feature Extraction**  
   - Feature descriptor: Histogram of Oriented Gradients (HOG)
   - Parameters:
     Pixels per cell: (4, 4)
     Cells per block: (2, 2)
   - HOG transforms each image into a vector of gradient features representing shape and stroke direction, crucial for handwritten character analysis.

5. **Model Configuration**  
   - Classifier: Support Vector Machine (SVC)
   - Kernel: Linear
   - Regularization (C): 0.1
     model = SVC(kernel='linear', C=0.1)

6. **Feature Extraction**  
   - Cross-validation: Leave-One-Out Cross Validation (LOOCV)
     loo = LeaveOneOut()
     y_pred = cross_val_predict(model, X_features, y_data, cv=loo, n_jobs=-1)
   - LOOCV ensures that every single sample is tested exactly once, providing a highly unbiased performance estimate.

---

## Results
1. **Accuracy**
The overall classification accuracy obtained was approximately : Accuracy 87.8462%
This shows that nearly 88% of the handwritten letters were correctly identified by the model.

2. **Classification Report**
   - Precision, measures correctness of positive predictions
   - Recall, measures how many actual positives were identified
   - F1-Score, harmonic mean of precision and recall
The classification report indicated high precision and recall across most letters, though some confusion occurred between similar-looking letters such as C ↔ G, I ↔ J, and U ↔ V.

3. **Confusion Matrix**
   The confusion matrix below illustrates the per-class prediction performance:
   - Diagonal elements represent correct classifications.
   - Off-diagonal elements indicate misclassifications.
   - Most predictions are tightly concentrated along the diagonal, showing strong model consistency.
   - Slight confusion observed for letters with similar shapes (e.g., O vs Q, U vs V, X vs Y).

## File Locations
Input Dataset: C:/Users/HP/assignment/4212301026_ATS VISIONMECHINE/archive/emnist-letters-train.csv

Generated Output:

[Confusion Matrix Image](4212301026_ATS VISIONMECHINE/confusion_matrix_loocv_belajar.png)

---

## Conclusion
- The combination of HOG features and SVM (linear kernel) achieved a high recognition accuracy of ~87.85% on the EMNIST Letters dataset.
- HOG effectively captured edge and shape information, which is essential for handwritten character discrimination.
- The SVM classifier provided good generalization and robustness, while LOOCV ensured reliable performance estimation.
- Minor misclassifications occurred for visually similar characters, suggesting that further improvement could be achieved using:
  - Nonlinear SVM kernels (e.g., RBF)
  - Data augmentation
  - Deeper feature extraction methods (e.g., CNN)