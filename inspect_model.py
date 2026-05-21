import pandas as pd
import joblib

raw_df = pd.read_csv('credit_risk_dataset.csv')
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

print('shape', raw_df.shape)
print('loan_status counts')
print(raw_df['loan_status'].value_counts(normalize=True))
print('loan_percent_income stats')
print(raw_df['loan_percent_income'].describe())

num_cols = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
print('scaler mean', scaler.mean_.tolist())
print('scaler var', scaler.var_.tolist())
print('scaler scale', scaler.scale_.tolist())

print('model intercept', model.intercept_.tolist())
print('feature len', len(feature_columns))
print('coeffs nonzero', sum(abs(model.coef_[0]) > 1e-6))
for col, coef in zip(feature_columns, model.coef_[0]):
    if abs(coef) > 0.1:
        print(col, coef)
