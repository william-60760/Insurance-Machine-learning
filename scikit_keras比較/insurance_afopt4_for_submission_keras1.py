import joblib
import pandas as pd
import numpy as np


def forSubmission(filename, method):
    model_pretrained = joblib.load(filename)

    test1_train1_df = pd.read_csv("test.csv")
    test1_train1_df.isnull().sum().sort_values(ascending=False)
    submissionid = test1_train1_df['id']
# 丟掉影響較小的欄位且刪除群組中有爭議的 Smoking status, Previous claims, Occupation
    test1_train1_df = test1_train1_df.drop(columns=['Policy Start Date', 'Exercise Frequency','id','Smoking Status','Previous Claims','Occupation'])

# 與訓練同樣的填補與欄位轉換（必要時調整）
    test1_train1_df['Credit Score'].fillna(test1_train1_df['Credit Score'].median(), inplace=True)
    test1_train1_df['Health Score'].fillna(test1_train1_df['Health Score'].median(), inplace=True)
    test1_train1_df['Annual Income'].fillna(test1_train1_df['Annual Income'].median(), inplace=True)
    test1_train1_df['Age'].fillna(test1_train1_df['Age'].median(), inplace=True)
    test1_train1_df['Vehicle Age'].fillna(test1_train1_df['Vehicle Age'].median(), inplace=True)
    test1_train1_df['Insurance Duration'].fillna(test1_train1_df['Insurance Duration'].median(), inplace=True)

# Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
    np.random.seed(100) # 設定隨機種子以確保結果可重現

    random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=test1_train1_df['Number of Dependents'].isnull().sum())
    null_nod_location = test1_train1_df['Number of Dependents'].isnull()
    test1_train1_df.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

    random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=test1_train1_df['Customer Feedback'].isnull().sum())
    null_cf_location = test1_train1_df['Customer Feedback'].isnull()
    test1_train1_df.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback

    random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=test1_train1_df['Marital Status'].isnull().sum())
    null_location = test1_train1_df['Marital Status'].isnull()
    test1_train1_df.loc[null_location, 'Marital Status'] = random_marital_status

    test1_train1_df['Customer Feedback Number'] = test1_train1_df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
    test1_train1_df['Gender Number'] = test1_train1_df['Gender'].map({'Male': 0, 'Female': 1})
    test1_train1_df['Marital Status Number'] = test1_train1_df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    test1_train1_df['Location Number'] = test1_train1_df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
    test1_train1_df['Policy Type Number'] = test1_train1_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
    test1_train1_df['Education Level Number'] = test1_train1_df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
    test1_train1_df['Property Type Number'] = test1_train1_df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})

# 丟棄與訓練時相同的欄位
    test1_train1_df = test1_train1_df.drop(columns=[ 'Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level', 'Property Type'])
    # predict
    predictions2 = model_pretrained.predict(test1_train1_df)

    # save to csv
    forSubmissiondf_test = pd.DataFrame(
        {"id": submissionid, 
        "Premium Amount": predictions2})
    forSubmissiondf_test.to_csv(f"Insurance_opt_{method}_forsubmission.csv", index=False)


# 放入檔案路徑，以及方法縮寫以便輸出時命名
forSubmission("Insurance-Linear Regression-mode.afopt4.pkl", "LR-AF4") 
forSubmission("Insurance-RF Regressor-mode.afopt4.pkl", "RFR-AF4") 

