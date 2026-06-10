from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Features and target
X = data.iloc[:, 2:4].values
y = data.iloc[:, 4].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Train SVM classifier
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)