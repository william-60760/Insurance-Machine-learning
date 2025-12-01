import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
df.head()
df.info()
df.isnull().sum().sort_values(ascending=False)
# 推斷影響較小（幫你丟掉id避免放入模型）
df = df.drop(columns=['Policy Start Date', 'Exercise Frequency','id'])
# 用中位數填補
df['Credit Score'].fillna(df['Credit Score'].median(), inplace=True)
df['Health Score'].fillna(df['Health Score'].median(), inplace=True)
df['Annual Income'].fillna(df['Annual Income'].median(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Vehicle Age'].fillna(df['Vehicle Age'].median(), inplace=True)
df['Insurance Duration'].fillna(df['Insurance Duration'].median(), inplace=True)
# 用0填補
df['Previous Claims'].fillna(0, inplace=True)
# Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
np.random.seed(100) # 設定隨機種子以確保結果可重現
random_occupation = np.random.choice(['Unemployed', 'Self-Employed', 'Employed'], size=df['Occupation'].isnull().sum())
null_location = df['Occupation'].isnull()
df.loc[null_location, 'Occupation'] = random_occupation

random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=df['Marital Status'].isnull().sum())
null_location = df['Marital Status'].isnull()
df.loc[null_location, 'Marital Status'] = random_marital_status

random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=df['Number of Dependents'].isnull().sum())
null_nod_location = df['Number of Dependents'].isnull()
df.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=df['Customer Feedback'].isnull().sum())
null_cf_location = df['Customer Feedback'].isnull()
df.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback
# 傳換成數值
df['Occupation Number'] = df['Occupation'].map({'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2})
df['Customer Feedback Number'] = df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
df['Gender Number'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Marital Status Number'] = df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
df['Location Number'] = df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
df['Policy Type Number'] = df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
df['Education Level Number'] = df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
df['Smoking Status Number'] = df['Smoking Status'].map({'No': 0, "Yes": 1})
df['Property Type Number'] = df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})
# 丟棄原本非數值
df = df.drop(columns=['Occupation', 'Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level', 'Smoking Status', 'Property Type'])


'''
上面訓練資料集預處理的程式基本都不做更動
除了固定隨機種子和丟棄id欄位外
'''
df.to_csv("bfopt_train_cleaned.csv", index=False) #用於輸出訓練資料