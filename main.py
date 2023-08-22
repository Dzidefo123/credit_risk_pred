from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

app = FastAPI()

# Load the saved combined model
model = joblib.load('combined_model.pkl')

df = pd.read_csv('cs-training.csv')

class CreditRiskInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float
    
@app.post("/predict/")
async def predict(credit_risk_input: CreditRiskInput):
    input_data = [
        credit_risk_input.RevolvingUtilizationOfUnsecuredLines,
        credit_risk_input.age,
        credit_risk_input.NumberOfTime30_59DaysPastDueNotWorse,
        credit_risk_input.DebtRatio,
        credit_risk_input.MonthlyIncome,
        credit_risk_input.NumberOfOpenCreditLinesAndLoans,
        credit_risk_input.NumberOfTimes90DaysLate,
        credit_risk_input.NumberRealEstateLoansOrLines,
        credit_risk_input.NumberOfTime60_89DaysPastDueNotWorse,
        credit_risk_input.NumberOfDependents
    ]
    probability_of_default = model.predict_proba([input_data])[0][1]  # Probability of class 1 (default)
    return {"probability_of_default": probability_of_default, **credit_risk_input.dict()}
