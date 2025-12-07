import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train1_df = pd.read_csv("train.csv")
train1_df.head()
train1_df.info()
train1_df.isnull().sum().sort_values(ascending=False)
# 推斷影響較小（幫你丟掉id避免放入模型）
train1_df = train1_df.drop(columns=['Policy Start Date', 'Exercise Frequency','id', 'Smoking Status', 'Previous Claims', 'Occupation'])
# 用中位數填補
train1_df['Credit Score'].fillna(train1_df['Credit Score'].median(), inplace=True)
train1_df['Health Score'].fillna(train1_df['Health Score'].median(), inplace=True)
train1_df['Annual Income'].fillna(train1_df['Annual Income'].median(), inplace=True)
train1_df['Age'].fillna(train1_df['Age'].median(), inplace=True)
train1_df['Vehicle Age'].fillna(train1_df['Vehicle Age'].median(), inplace=True)
train1_df['Insurance Duration'].fillna(train1_df['Insurance Duration'].median(), inplace=True)

# Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
np.random.seed(100) # 設定隨機種子以確保結果可重現

random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=train1_df['Marital Status'].isnull().sum())
null_location = train1_df['Marital Status'].isnull()
train1_df.loc[null_location, 'Marital Status'] = random_marital_status

random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=train1_df['Number of Dependents'].isnull().sum())
null_nod_location = train1_df['Number of Dependents'].isnull()
train1_df.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=train1_df['Customer Feedback'].isnull().sum())
null_cf_location = train1_df['Customer Feedback'].isnull()
train1_df.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback
# 傳換成數值

train1_df['Customer Feedback Number'] = train1_df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
train1_df['Gender Number'] = train1_df['Gender'].map({'Male': 0, 'Female': 1})
train1_df['Marital Status Number'] = train1_df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
train1_df['Location Number'] = train1_df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
train1_df['Policy Type Number'] = train1_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
train1_df['Education Level Number'] = train1_df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
train1_df['Property Type Number'] = train1_df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})
# 丟棄原本非數值
train1_df = train1_df.drop(columns=[ 'Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level',  'Property Type'])

# 定義模型
np.random.seed(42)
dataset1 = train1_df.values
np.random.shuffle(dataset1)

train1_df.to_csv("train_preprocessed_.csv", index=False)