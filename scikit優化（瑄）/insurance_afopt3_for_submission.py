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

    #優化2-特徵工程 (Feature Engineering)
    # 1. 對數轉換 (處理偏態，使用 log1p = log(x+1))
    df_test['Annual Income_log'] = np.log1p(df_test['Annual Income']) 
    df_test['Previous Claims_log'] = np.log1p(df_test['Previous Claims'])

    # 2. 創造交互特徵 (Interaction Features)
    df_test['Income_VehicleAge'] = df_test['Annual Income'] * df_test['Vehicle Age']
    df_test['Health_Smoker_Risk'] = df_test['Health Score'] * df_test['Smoking Status'].map({'No': 0, "Yes": 1}) 
    # 這裡需要用到 Smoking Status 的原始類別值，因此先做映射

    # 3. 年齡分組 (Discretization)
    # 根據數據的 min/max 設置分組邊界
    bins_train = [df_train['Age'].min(), 30, 45, 60, df_train['Age'].max() + 1] 
    labels = [1, 2, 3, 4]
    # 將 Age 分組並轉換為數字編碼
    df_test['Age_Group_Number'] = pd.cut(df_test['Age'], bins=bins_train, labels=labels, right=False).astype(int)

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
    # 丟棄原始非數值欄位，以及被新特徵取代的原始數值欄位
    df_test = df_test.drop(columns=[
    'Occupation', 'Customer Feedback', 'Gender', 'Marital Status', 
    'Location', 'Policy Type', 'Education Level', 'Smoking Status', 
    'Property Type', 'Annual Income', 'Previous Claims', 'Age' # 刪除被新的特徵工程取代的原始欄位
])

    df_test = df_test[model_pretrained.feature_names_in_]

    # predict
    predictions2 = model_pretrained.predict(df_test)

    # save to csv
    forSubmissiondf_test = pd.DataFrame(
        {"id": submissionid, 
        "Premium Amount": predictions2})
    forSubmissiondf_test.to_csv(f"Insurance_opt_{method}_for_submission.csv", index=False)


# 放入檔案路徑，以及方法縮寫以便輸出時命名
forSubmission("Insurance-Linear Regression-mode.afopt3.pkl", "LR-AF3") 
forSubmission("Insurance-RF Regressor-mode.afopt3.pkl", "RFR-AF3") 