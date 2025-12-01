
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
    # Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
    np.random.seed(100) # 設定隨機種子以確保結果可重現
    random_occupation = np.random.choice(['Unemployed', 'Self-Employed', 'Employed'], size=df_test['Occupation'].isnull().sum())
    null_location = df_test['Occupation'].isnull()
    df_test.loc[null_location, 'Occupation'] = random_occupation

    random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=df_test['Marital Status'].isnull().sum())
    null_location = df_test['Marital Status'].isnull()
    df_test.loc[null_location, 'Marital Status'] = random_marital_status

    random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=df_test['Number of Dependents'].isnull().sum())
    null_nod_location = df_test['Number of Dependents'].isnull()
    df_test.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

    random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=df_test['Customer Feedback'].isnull().sum())
    null_cf_location = df_test['Customer Feedback'].isnull()
    df_test.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback
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
    forSubmissiondf_test.to_csv(f"Insurance_bfopt_{method}_for_submission.csv", index=False)


# 放入檔案路徑，以及方法縮寫以便輸出時命名
forSubmission("Insurance-Linear Regression-bfopt.pkl", "LR") 
forSubmission("Insurance-RF Regressor-bfopt.pkl", "RFR") 