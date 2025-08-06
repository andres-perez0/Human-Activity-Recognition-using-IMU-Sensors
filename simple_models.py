# custom libaries
from dataCtrl import dataCtrl

# for simple/pre-built models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error

# for number - crunching
import numpy as np

DataProcessor=dataCtrl()

data = DataProcessor.initialize_data()
window_size=250
overlap_percentage=0.3
train_data, test_data, train_labels, test_labels = DataProcessor.preprocess_and_spilt(data, window_size, overlap_percentage,0.2)

X_train_flat = np.array([window.reshape(-1) for window in train_data])
X_test_flat = np.array([window.reshape(-1) for window in test_data])

# Linear Regreesion
lr_model = LinearRegression()
lr_model.fit(X_train_flat, train_labels)
lr_predictions = lr_model.predict(X_test_flat)

# To calculate 'accuracy', we need to convert predictions to class labels.
# We'll use a threshold of 0.5.
lr_predictions_classes = (lr_predictions > 0.5).astype(int)
lr_accuracy = accuracy_score(test_labels, lr_predictions_classes)
lr_mse = mean_squared_error(test_labels, lr_predictions)

print(f"Linear Regression 'Accuracy' (with 0.5 threshold): {lr_accuracy*100:.4f}")
print(f"Linear Regression Mean Squared Error: {lr_mse*100:.4f}")

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_flat, train_labels)
svm_predictions = svm_model.predict(X_test_flat)
svm_accuracy = accuracy_score(test_labels, svm_predictions)

print(f"Support Vector Classifier (SVC) Accuracy: {svm_accuracy*100:.4f}")

iterations=10
rf_accuracy_list = []

for i in range(iterations):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_model.fit(X_train_flat, train_labels)
    rf_predicitions = rf_model.predict(X_test_flat)
    rf_accuracy = accuracy_score(test_labels, rf_predicitions)
    rf_accuracy_list.append(rf_accuracy)

print(f"Random Forest Classifier Accuracy:")
[print(f'{i+1}. rf_accuracy : {val*100:.2f} %') for i, val in enumerate(rf_accuracy_list)]

