# task-10
AI & ML Internship – Task 10
KNN – Handwritten Digit Classification
________________________________________
1. Introduction
Handwritten digit classification is a supervised machine learning problem where the goal is to correctly identify digits (0–9) from image data.
In this task, we use the K-Nearest Neighbors (KNN) algorithm, which is a distance-based classifier, to classify digits from the Sklearn Digits dataset.
________________________________________
2. Tools Used
•	Python
•	Scikit-learn
•	Matplotlib
________________________________________
3. Dataset
•	Primary Dataset: Sklearn Digits dataset (load_digits())
•	Each digit image is of size 8 × 8 pixels
•	Total features per image = 64
________________________________________
4. Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
________________________________________
5. Load Digits Dataset
digits = load_digits()
X = digits.data
y = digits.target

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
________________________________________
6. Visualize Sample Digit Images
plt.figure(figsize=(8,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.show()
Purpose:
To visually confirm that digit images match their labels.
________________________________________
7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
________________________________________
8. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Reason:
KNN uses distance calculations, so feature scaling is required to avoid bias.
________________________________________
9. Train KNN Model (K = 3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with K=3:", accuracy)
________________________________________
10. Try Multiple K Values
k_values = [3, 5, 7, 9]
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

print("Accuracies:", accuracies)
________________________________________
11. Accuracy vs K Plot
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()
Purpose:
To select the best K value based on model performance.
________________________________________
12. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
Purpose:
To identify which digits are misclassified.
________________________________________
13. Display Test Images with Predictions
plt.figure(figsize=(8,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i].reshape(8,8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.show()

14. Final Outcome
•	Learned distance-based classification
•	Understood importance of feature scaling
•	Tuned K value for better accuracy
•	Evaluated model using accuracy and confusion matrix

15. Interview Questions – Short Answers
1. What is K in KNN?
K is the number of nearest neighbors considered for classification.
2. Why is scaling required for KNN?
Because KNN uses distance calculations and unscaled features can dominate distances.
3. What is Euclidean distance?
It is the straight-line distance between two points in feature space.
4. What happens if K is too low?
The model becomes sensitive to noise and may overfit.
5. What are limitations of KNN?
High computation cost, slow prediction, and sensitivity to irrelevant features.


