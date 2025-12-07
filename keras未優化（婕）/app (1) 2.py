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

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, Dense


def rmse(y_true, y_pred):
    # Keras/TF 的函式庫操作
    squared_error = tf.square(y_pred - y_true)
    mse = tf.reduce_mean(squared_error)
    return tf.sqrt(mse)

def draw_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "r", label="Validation Loss")
        plt.title("Training and Validation Loss")
    else:
        plt.title("Training Loss")
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def draw_metric(history):
    #訓練集RMSE
    metric = history.history["rmse"]
    epochs = range(1, len(metric) + 1)
    #檢查驗證集RMSE是否存在
    if "val_rmse" in history.history:
        # 提取驗證集RMSE
        val_metric = history.history["val_rmse"]
        plt.plot(epochs, val_metric, "r--", label="Validation RMSE")
        plt.title("Training and Validation RMSE")
    else:
        # 否則只繪製訓練RMSE
        plt.title("Training RMSE")

    plt.plot(epochs, metric, "b-", label="Training RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE (Error)")
    plt.legend()
    plt.show()


np.random.seed(42)
dataset = df.values
np.random.shuffle(dataset)

# 正確選取第0-8 與 9-17
X = dataset[:, np.r_[0:8, 9:]]
Y = dataset[:, 8]

model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mae', metrics=['mae', rmse])

history = model.fit(X, Y, validation_split=0.2, epochs=50, batch_size=100)
print("----------------------------------")

draw_loss(history)
draw_metric(history)

# --- evaluate on test.csv (必須與訓練時同樣的前處理) ---
test_df = pd.read_csv("test.csv")
test_df.isnull().sum().sort_values(ascending=False)
# 丟掉影響較小的欄位
test_df = test_df.drop(columns=['Policy Start Date', 'Exercise Frequency','id'])

# 與訓練同樣的填補與欄位轉換（必要時調整）
test_df['Credit Score'].fillna(test_df['Credit Score'].median(), inplace=True)
test_df['Health Score'].fillna(test_df['Health Score'].median(), inplace=True)
test_df['Annual Income'].fillna(test_df['Annual Income'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Vehicle Age'].fillna(test_df['Vehicle Age'].median(), inplace=True)
test_df['Insurance Duration'].fillna(test_df['Insurance Duration'].median(), inplace=True)
test_df['Previous Claims'].fillna(0, inplace=True)

# Marital Status, Number of Dependents, Customer Feedback只有特定幾個值且分布平均，用隨機值填補
np.random.seed(100) # 設定隨機種子以確保結果可重現

random_occupation = np.random.choice(['Unemployed', 'Self-Employed', 'Employed'], size=test_df['Occupation'].isnull().sum())
null_location = test_df['Occupation'].isnull()
test_df.loc[null_location, 'Occupation'] = random_occupation

random_number_of_dependents = np.random.choice([0, 1, 2, 3, 4], size=test_df['Number of Dependents'].isnull().sum())
null_nod_location = test_df['Number of Dependents'].isnull()
test_df.loc[null_nod_location, 'Number of Dependents'] = random_number_of_dependents

random_customer_feedback = np.random.choice(['Poor', 'Average', 'Good'], size=test_df['Customer Feedback'].isnull().sum())
null_cf_location = test_df['Customer Feedback'].isnull()
test_df.loc[null_cf_location, 'Customer Feedback'] = random_customer_feedback

random_marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=test_df['Marital Status'].isnull().sum())
null_location = test_df['Marital Status'].isnull()
test_df.loc[null_location, 'Marital Status'] = random_marital_status

test_df['Occupation Number'] = test_df['Occupation'].map({'Unemployed': 0, 'Self-Employed': 1, 'Employed': 2})
test_df['Customer Feedback Number'] = test_df['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
test_df['Gender Number'] = test_df['Gender'].map({'Male': 0, 'Female': 1})
test_df['Marital Status Number'] = test_df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
test_df['Location Number'] = test_df['Location'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
test_df['Policy Type Number'] = test_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium': 2})
test_df['Education Level Number'] = test_df['Education Level'].map({'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3})
test_df['Smoking Status Number'] = test_df['Smoking Status'].map({'No': 0, "Yes": 1})
test_df['Property Type Number'] = test_df['Property Type'].map({'House': 0, 'Condo': 1, 'Apartment': 2})


# 丟棄與訓練時相同的欄位
test_df = test_df.drop(columns=['Occupation', 'Customer Feedback', 'Gender', 'Marital Status', 'Location', 'Policy Type', 'Education Level', 'Smoking Status', 'Property Type'])
test_df.head()

test_data = test_df.values
X_test = test_data[:, 0:]
Y_test = model.predict(X_test, verbose=0)


submission_df = pd.DataFrame({"id": np.arange(1200000, 1200000 + len(Y_test)), "Premium Amount": Y_test.flatten()})
submission_df.to_csv("bfkeras_submission.csv", index=False)