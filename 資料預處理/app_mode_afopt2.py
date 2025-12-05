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

#優化2-特徵工程 (Feature Engineering)
# 1. 對數轉換 (處理偏態，使用 log1p = log(x+1))
df['Annual Income_log'] = np.log1p(df['Annual Income']) 
df['Previous Claims_log'] = np.log1p(df['Previous Claims'])

# 2. 創造交互特徵 (Interaction Features)
df['Income_VehicleAge'] = df['Annual Income'] * df['Vehicle Age']
df['Health_Smoker_Risk'] = df['Health Score'] * df['Smoking Status'].map({'No': 0, "Yes": 1}) 
# 這裡需要用到 Smoking Status 的原始類別值，因此先做映射

# 3. 年齡分組 (Discretization)
# 根據數據的 min/max 設置分組邊界
bins = [df['Age'].min(), 30, 45, 60, df['Age'].max() + 1] # +1 確保包含最大值
labels = [1, 2, 3, 4]
# 將 Age 分組並轉換為數字編碼
df['Age_Group_Number'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False).astype(int)

# ==========================================================
#數值編碼與丟棄舊欄位 (保留新的交互和轉換特徵)
# ==========================================================

# 傳換成數值 (保持不變)
df['Occupation Number'] = df['Occupation'].map({'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2})
df['Customer Feedback Number'] = df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
df['Gender Number'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Marital Status Number'] = df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
df['Location Number'] = df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
df['Policy Type Number'] = df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
df['Education Level Number'] = df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
df['Smoking Status Number'] = df['Smoking Status'].map({'No': 0, "Yes": 1})
df['Property Type Number'] = df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})


# 丟棄原始非數值欄位，以及被新特徵取代的原始數值欄位
df = df.drop(columns=[
    'Occupation', 'Customer Feedback', 'Gender', 'Marital Status', 
    'Location', 'Policy Type', 'Education Level', 'Smoking Status', 
    'Property Type', 'Annual Income', 'Previous Claims', 'Age' # 刪除被新的特徵工程取代的原始欄位
])