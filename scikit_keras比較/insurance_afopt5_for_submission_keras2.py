import joblib
import pandas as pd
import numpy as np


def forSubmission(filename, method):
    model_pretrained = joblib.load(filename)

    test2_train2_df = pd.read_csv("test.csv")
    test2_train2_df.isnull().sum().sort_values(ascending=False)
    submissionid = test2_train2_df['id']
# 丟掉影響較小的欄位
    test2_train2_df = test2_train2_df.drop(columns=['Policy Start Date', 'Exercise Frequency','id'])

# 與訓練同樣的填補與欄位轉換（必要時調整）
    test2_train2_df['Credit Score'].fillna(test2_train2_df['Credit Score'].median(), inplace=True)
    test2_train2_df['Health Score'].fillna(test2_train2_df['Health Score'].median(), inplace=True)
    test2_train2_df['Annual Income'].fillna(test2_train2_df['Annual Income'].median(), inplace=True)
    test2_train2_df['Age'].fillna(test2_train2_df['Age'].median(), inplace=True)
    test2_train2_df['Vehicle Age'].fillna(test2_train2_df['Vehicle Age'].median(), inplace=True)
    test2_train2_df['Insurance Duration'].fillna(test2_train2_df['Insurance Duration'].median(), inplace=True)

# Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
    np.random.seed(100) # 設定隨機種子以確保結果可重現

    random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=test2_train2_df['Number of Dependents'].isnull().sum())
    null_nod_location = test2_train2_df['Number of Dependents'].isnull()
    test2_train2_df.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

    random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=test2_train2_df['Customer Feedback'].isnull().sum())
    null_cf_location = test2_train2_df['Customer Feedback'].isnull()
    test2_train2_df.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback

    random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=test2_train2_df['Marital Status'].isnull().sum())
    null_location = test2_train2_df['Marital Status'].isnull()
    test2_train2_df.loc[null_location, 'Marital Status'] = random_marital_status

    test2_train2_df['Customer Feedback Number'] = test2_train2_df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
    test2_train2_df['Gender Number'] = test2_train2_df['Gender'].map({'Male': 0, 'Female': 1})
    test2_train2_df['Marital Status Number'] = test2_train2_df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    test2_train2_df['Location Number'] = test2_train2_df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
    test2_train2_df['Policy Type Number'] = test2_train2_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
    test2_train2_df['Education Level Number'] = test2_train2_df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
    test2_train2_df['Property Type Number'] = test2_train2_df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})

# 丟棄與訓練時相同的欄位
    test2_train2_df= test2_train2_df.drop(columns=[ 'Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level', 'Property Type'])

    test2_train2_df = test2_train2_df[model_pretrained.feature_names_in_]

    # predict
    predictions2 = model_pretrained.predict(test2_train2_df)

    # save to csv
    forSubmissiondf_test = pd.DataFrame(
        {"id": submissionid, 
        "Premium Amount": predictions2})
    forSubmissiondf_test.to_csv(f"Insurance_opt_{method}_forsubmission.csv", index=False)


# 放入檔案路徑，以及方法縮寫以便輸出時命名
forSubmission("Insurance-Linear Regression-mode.afopt5.pkl", "LR-AF5") 
forSubmission("Insurance-RF Regressor-mode.afopt5.pkl", "RFR-AF5") 

