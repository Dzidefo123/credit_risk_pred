# credit_risk_pred
Credit risk prediction using xgboost and neural network with fastAPI 

Credit Risk Prediction Documentation
Project Overview

This document provides an overview of the credit risk prediction project, including data preprocessing, modeling, and the development of a web application for credit risk prediction.
Data Source  https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

The dataset for this project was obtained from Kaggle and focuses on credit risk prediction. The dataset contains information about various features related to credit applicants, including demographic and financial attributes. The goal is to predict the likelihood of credit default based on the provided features.

Dataset Source: Link to Kaggle Dataset
Data Preprocessing
Data Cleaning

The raw dataset contained missing values, which were addressed using the following steps:

    Dropped the 'Unnamed: 0' column, if present, as it appeared to be an unnecessary index column.
    Checked for duplicate records and removed any duplicates to ensure data integrity.
    Calculated the percentage of missing values for each feature and decided on handling strategies.

Handling Missing Values

    'MonthlyIncome' and 'NumberOfDependents' had missing values. The 'MonthlyIncome' column had significant missing values, and the 'NumberOfDependents' column had a smaller percentage of missing values.
    For 'MonthlyIncome':
        Records with missing 'MonthlyIncome' were divided into two subsets: those with missing 'NumberOfDependents' and those with non-missing 'NumberOfDependents'.
        For records with non-missing 'NumberOfDependents', the missing 'MonthlyIncome' values were filled with the median value of 'MonthlyIncome'.
        For records with missing 'NumberOfDependents', the missing 'MonthlyIncome' values were also filled with the median value of 'MonthlyIncome'.
    For 'NumberOfDependents', records with missing values were filled with the mode value, which was '0'.

Modeling Approach

Two machine learning models were used for credit risk prediction: an XGBoost model and a Neural Network model. The predictions from both models were averaged to obtain the final credit risk prediction.

    XGBoost Model:
        An XGBoost classifier was trained on the preprocessed dataset.
        The model was trained to predict the likelihood of credit default based on the provided features.
        The XGBoost model was chosen due to its effectiveness in handling imbalanced datasets and providing probability predictions.

    Neural Network Model:
        A neural network was designed using TensorFlow/Keras.
        The neural network architecture included multiple hidden layers with activation functions.
        The model was trained using the same preprocessed dataset to predict credit default probability.

Web Application Development

A web application was developed using FastAPI to provide an interactive platform for users to predict credit risk.

    FastAPI:
        FastAPI was chosen as the web framework due to its ease of use and support for asynchronous operations.
        Endpoints were defined to handle user input and provide predictions.

    Web Interface:
        The web interface included a form where users could input their feature values for credit risk prediction.
        The input data was sent to the FastAPI backend for processing.

    Final Prediction:
        The backend processed the user input using the trained models (XGBoost and Neural Network).
        The probabilities predicted by both models were averaged to obtain the final credit risk prediction.

Conclusion

The credit risk prediction project involved data preprocessing, model training, and the development of a web application using FastAPI. The project aimed to provide users with a convenient way to assess credit risk and make informed decisions.

