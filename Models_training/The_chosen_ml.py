from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

cleaned_dt = pd.read_excel('C:\\Users\\Nour El Houda\\Documents\\GitHub\\pediatric-appendicitis-diagnosis\\data\\cleaned_data.xlsx', engine='openpyxl')
cleaned_dt.fillna(0, inplace=True)
cleaned_dt['Diagnosis'].replace({'appendicitis' : 1, 'no appendicitis' : 0}, inplace=True)

cleaned_dt.columns = cleaned_dt.columns.str.replace(r'\s+', '_', regex=True)  
cleaned_dt.columns = cleaned_dt.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  

X = cleaned_dt.drop(columns=['Diagnosis'])  
y = cleaned_dt['Diagnosis']


X = pd.get_dummies(X, drop_first=True)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normilizing the values 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

categorical_columns = cleaned_dt.select_dtypes(include=['object']).columns

label_encoders = {}
for column in categorical_columns:
    if cleaned_dt[column].dtype == 'object':  
        encoder = LabelEncoder()
        cleaned_dt[column] = encoder.fit_transform(cleaned_dt[column].fillna('Unknown')) 
        label_encoders[column] = encoder

X = cleaned_dt.drop(columns=['Diagnosis'])
y = cleaned_dt['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred_lgb)
roc_auc = roc_auc_score(y_test, y_pred_proba_lgb)
precision = precision_score(y_test, y_pred_lgb)
recall = recall_score(y_test, y_pred_lgb)
f1 = f1_score(y_test, y_pred_lgb)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")