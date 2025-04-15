# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = 'D:/Park_Pred_Model/brim_data.csv'
df = pd.read_csv(file_path)
print(df.head())

# Preprocessing
df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
df['hour'] = df['LastUpdated'].dt.hour
df['day_of_week'] = df['LastUpdated'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Target column
df['Availability'] = (df['Occupancy'] < 0.9 * df['Capacity']).astype(int)

# Encode categorical feature
le = LabelEncoder()
df['SystemCode_encoded'] = le.fit_transform(df['SystemCodeNumber'])

# Features and target
features = ['SystemCode_encoded', 'hour', 'day_of_week', 'is_weekend']
X = df[features]
y = df['Availability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Apply SMOTE to balance data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Random Forest with class_weight to handle imbalance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluation on test set
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nTest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Evaluation on training set
y_train_pred = rf_model.predict(X_train)
print("\nRandom Forest Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nTraining Classification Report:\n", classification_report(y_train, y_train_pred))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

sns.countplot(x=y_train)
plt.title('Class Distribution After SMOTE (Training Data)')
plt.xlabel('Availability')
plt.ylabel('Count')
plt.show()


# Visualization: Average Occupancy by Hour
avg_occ = df.groupby('hour')['Occupancy'].mean()
avg_occ.plot(kind='line', marker='o')
plt.title('Average Occupancy by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Occupancy')
plt.grid(True)
plt.show()

# Visualization: Availability by Location
availability_by_location = df.groupby('SystemCodeNumber')['Availability'].mean().sort_values()
availability_by_location.plot(kind='barh', figsize=(10, 6), color='skyblue')
plt.title('Average Availability by Parking Location')
plt.xlabel('Availability Rate')
plt.ylabel('Parking Location')
plt.show()

# Visualization: Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='green')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Relative Importance')
plt.show()

# Visualization: Confusion Matrix (Test)
conf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Random Forest (Test)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save trained model
with open('D:/Park_Pred_Model/rf_parking_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save LabelEncoder
with open('D:/Park_Pred_Model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
