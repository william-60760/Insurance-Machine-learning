import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
df.head()
df.info()
df.isnull().sum().sort_values(ascending=False)
# 推斷影響較小
df = df.drop(columns=['Policy Start Date', 'Exercise Frequency'])
# 用中位數填補
df['Credit Score'].fillna(df['Credit Score'].median(), inplace=True)
df['Health Score'].fillna(df['Health Score'].median(), inplace=True)
df['Annual Income'].fillna(df['Annual Income'].median(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Vehicle Age'].fillna(df['Vehicle Age'].median(), inplace=True)
df['Insurance Duration'].fillna(df['Insurance Duration'].median(), inplace=True)
# 用0填補
df['Previous Claims'].fillna(0, inplace=True)

# 優化-眾數填補 (Mode Imputation)

# 1. Occupation
mode_occupation = df['Occupation'].mode()[0]
df['Occupation'].fillna(mode_occupation, inplace=True)

# 2. Marital Status
mode_marital = df['Marital Status'].mode()[0]
df['Marital Status'].fillna(mode_marital, inplace=True)

# 3. Number of Dependents (雖然是數值，但屬於類別型填補)
mode_dependents = df['Number of Dependents'].mode()[0]
df['Number of Dependents'].fillna(mode_dependents, inplace=True)

# 4. Customer Feedback
mode_feedback = df['Customer Feedback'].mode()[0]
df['Customer Feedback'].fillna(mode_feedback, inplace=True)

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
df.to_csv("mode_afopt_train_cleaned.csv", index=False) #用於輸出訓練資料