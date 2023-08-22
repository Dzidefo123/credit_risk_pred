from fastapi import FastAPI, Form, Request, Depends
from fastapi.templating import Jinja2Templates
import joblib
from pydantic import BaseModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import templates

app = FastAPI()

# Load the combined model
combined_model = joblib.load('combined_model.pkl')

# Load templates directory for Jinja2
templates = Jinja2Templates(directory="templates")

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


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict/")
async def predict(request: Request, credit_risk_input: CreditRiskInput):
    input_data = [credit_risk_input.RevolvingUtilizationOfUnsecuredLines,
                  credit_risk_input.age,
                  credit_risk_input.NumberOfTime30_59DaysPastDueNotWorse,
                  credit_risk_input.DebtRatio,
                  credit_risk_input.MonthlyIncome,
                  credit_risk_input.NumberOfOpenCreditLinesAndLoans,
                  credit_risk_input.NumberOfTimes90DaysLate,
                  credit_risk_input.NumberRealEstateLoansOrLines,
                  credit_risk_input.NumberOfTime60_89DaysPastDueNotWorse,
                  credit_risk_input.NumberOfDependents] 
    prediction = combined_model.predict([input_data])[0]
    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})

    
