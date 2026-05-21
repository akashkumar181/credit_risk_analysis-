import pandas as pd
import joblib

model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

input_data = {
    'person_age': 30,
    'person_income': 3000,
    'person_emp_length': 5,
    'loan_amnt': 100,
    'loan_int_rate': 9.97,
    'loan_percent_income': 19.96,
    'cb_person_cred_hist_length': 10.0,
    'person_home_ownership': 'MORTGAGE',
    'loan_intent': 'EDUCATION',
    'loan_grade': 'A',
    'cb_person_default_on_file': 'Y'
}

input_df = pd.DataFrame([input_data])
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
for c in feature_columns:
    if c not in input_encoded.columns:
        input_encoded[c] = 0
input_encoded = input_encoded[feature_columns]

num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

pred = model.predict(input_encoded)[0]
proba = model.predict_proba(input_encoded)[0]
print('pred', pred)
print('proba', proba)
print('proba_pred', proba[pred])
print('input_encoded', input_encoded.iloc[0].to_dict())
