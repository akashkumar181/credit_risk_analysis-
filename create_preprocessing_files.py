import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('credit_risk_dataset.csv')

numerical_features = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

categorical_features = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

scaler = StandardScaler()
scaler.fit(df_encoded[numerical_features])

# Save scaler and feature column order
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(df_encoded.drop('loan_status', axis=1).columns.tolist(), 'feature_columns.pkl')
print('Created scaler.pkl and feature_columns.pkl')
