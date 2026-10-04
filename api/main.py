"""Checkout entry point; packaged application lives in credit_risk.api."""
from credit_risk.api.main import create_app

app=create_app()
