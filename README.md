# Credit Risk Analyzer

This project is a beginner-friendly end-to-end credit risk analyzer built in a Jupyter Notebook and deployed with Streamlit.

## Project structure

- `credit_risk_analyzer.ipynb`: Notebook with EDA, preprocessing, model training, evaluation, and PyTorch demo.
- `streamlit_app.py`: Simple Streamlit app for making credit risk predictions.
- `credit_risk_dataset.csv`: Dataset used for training and analysis.
- `logistic_regression_model.pkl`: Saved Logistic Regression model.
- `random_forest_model.pkl`: Saved Random Forest model.
- `scaler.pkl`: Saved scaler for numerical preprocessing.
- `feature_columns.pkl`: Saved feature order for Streamlit preprocessing.
- `.github/workflows/ci.yml`: GitHub Actions workflow for CI.
- `requirements.txt`: Python dependencies.

## How to run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the notebook in Jupyter:
```bash
jupyter notebook credit_risk_analyzer.ipynb
```
3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Docker deployment

Build the Docker image from the repository root:
```bash
docker build -t credit-risk-analyzer .
```

Run the container locally:
```bash
docker run -p 8501:8501 credit-risk-analyzer
```

Then open:
```bash
http://localhost:8501
```

## Git & GitHub

1. Initialize Git (if not already initialized):
```bash
git init
```
2. Add files:
```bash
git add .
```
3. Commit changes:
```bash
git commit -m "Add credit risk analyzer project"
```
4. Push to GitHub:
```bash
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## CI/CD

The GitHub Actions workflow installs dependencies and runs a Python syntax check on `streamlit_app.py` for each push or pull request to `main`.
