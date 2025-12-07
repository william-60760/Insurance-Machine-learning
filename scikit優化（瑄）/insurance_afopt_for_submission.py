
import joblib
import pandas as pd
import numpy as np


def forSubmission(filename, method):
    model_pretrained = joblib.load(filename)

    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")
    #df_test_test.info()#用於觀察資訊

    submissionid = df_test['id']  # 預留id欄位以便最後輸出
    '''
    開始對測試資料前處理，需要和訓練集一樣的步驟！！！
    '''
    df_test = df_test.drop(columns=['Policy Start Date', 'Exercise Frequency','id'])
    # 用訓練資料的中位數填補
    df_test['Credit Score'].fillna(df_train['Credit Score'].median(), inplace=True)
    df_test['Health Score'].fillna(df_train['Health Score'].median(), inplace=True)
    df_test['Annual Income'].fillna(df_train['Annual Income'].median(), inplace=True)
    df_test['Age'].fillna(df_train['Age'].median(), inplace=True)
    df_test['Vehicle Age'].fillna(df_train['Vehicle Age'].median(), inplace=True)
    df_test['Insurance Duration'].fillna(df_train['Insurance Duration'].median(), inplace=True)
    # 用0填補
    df_test['Previous Claims'].fillna(0, inplace=True)

    #優化-眾數填補
    # 1. Occupation
    mode_occupation = df_train['Occupation'].mode()[0]
    df_test['Occupation'].fillna(mode_occupation, inplace=True)

    # 2. Marital Status
    mode_marital_status = df_train['Marital Status'].mode()[0]
    df_test['Marital Status'].fillna(mode_marital_status, inplace=True)

    # 3. Number of Dependents
    mode_nod = df_train['Number of Dependents'].mode()[0]
    df_test['Number of Dependents'].fillna(mode_nod, inplace=True)

    # 4. Customer Feedback
    mode_customer_feedback = df_train['Customer Feedback'].mode()[0]
    df_test['Customer Feedback'].fillna(mode_customer_feedback, inplace=True)

    # 傳換成數值
    df_test['Occupation Number'] = df_test['Occupation'].map({'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2})
    df_test['Customer Feedback Number'] = df_test['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
    df_test['Gender Number'] = df_test['Gender'].map({'Male': 0, 'Female': 1})
    df_test['Marital Status Number'] = df_test['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    df_test['Location Number'] = df_test['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
    df_test['Policy Type Number'] = df_test['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
    df_test['Education Level Number'] = df_test['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
    df_test['Smoking Status Number'] = df_test['Smoking Status'].map({'No': 0, "Yes": 1})
    df_test['Property Type Number'] = df_test['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})
    # 丟棄原本非數值
    df_test = df_test.drop(columns=['Occupation', 'Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level', 'Smoking Status', 'Property Type'])

    # predict
    predictions2 = model_pretrained.predict(df_test)

    # save to csv
    forSubmissiondf_test = pd.DataFrame(
        {"id": submissionid, 
        "Premium Amount": predictions2})
    forSubmissiondf_test.to_csv(f"Insurance_opt_{method}_for_submission.csv", index=False)


# 放入檔案路徑，以及方法縮寫以便輸出時命名
forSubmission("Insurance-Linear Regression-mode.afopt.pkl", "LR-AF1") 
forSubmission("Insurance-RF Regressor-mode.afopt.pkl", "RFR-AF1") 