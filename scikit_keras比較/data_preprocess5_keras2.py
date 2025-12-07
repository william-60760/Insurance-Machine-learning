import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train2_df = pd.read_csv("train.csv")
train2_df.head()
train2_df.info()
train2_df.isnull().sum().sort_values(ascending=False)
# 推斷影響較小（幫你丟掉id避免放入模型）
train2_df = train2_df.drop(columns=['Policy Start Date', 'Exercise Frequency','id','Smoking Status', 'Previous Claims', 'Occupation'])
# 用中位數填補
train2_df['Credit Score'].fillna(train2_df['Credit Score'].median(), inplace=True)
train2_df['Health Score'].fillna(train2_df['Health Score'].median(), inplace=True)
train2_df['Annual Income'].fillna(train2_df['Annual Income'].median(), inplace=True)
train2_df['Age'].fillna(train2_df['Age'].median(), inplace=True)
train2_df['Vehicle Age'].fillna(train2_df['Vehicle Age'].median(), inplace=True)
train2_df['Insurance Duration'].fillna(train2_df['Insurance Duration'].median(), inplace=True)

# Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
np.random.seed(100) # 設定隨機種子以確保結果可重現

random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=train2_df['Marital Status'].isnull().sum())
null_location = train2_df['Marital Status'].isnull()
train2_df.loc[null_location, 'Marital Status'] = random_marital_status

random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=train2_df['Number of Dependents'].isnull().sum())
null_nod_location = train2_df['Number of Dependents'].isnull()
train2_df.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=train2_df['Customer Feedback'].isnull().sum())
null_cf_location = train2_df['Customer Feedback'].isnull()
train2_df.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback
# 傳換成數值
train2_df['Customer Feedback Number'] = train2_df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
train2_df['Gender Number'] = train2_df['Gender'].map({'Male': 0, 'Female': 1})
train2_df['Marital Status Number'] = train2_df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
train2_df['Location Number'] = train2_df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
train2_df['Policy Type Number'] = train2_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
train2_df['Education Level Number'] = train2_df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
train2_df['Property Type Number'] = train2_df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})
# 丟棄原本非數值
train2_df = train2_df.drop(columns=['Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level', 'Property Type'])

np.random.seed(42)
# 標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale_cols = ['Annual Income', 'Health Score', 'Credit Score']
train2_df[scale_cols] = scaler.fit_transform(train2_df[scale_cols])
dataset2 = train2_df.values
np.random.shuffle(dataset2)
train2_df.to_csv("train_preprocessed_.csv", index=False)