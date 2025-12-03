import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

target = "Premium Amount"
feature = [
    "Age", "Annual Income", "Number of Dependents", "Health Score", "Vehicle Age", "Credit Score", "Insurance Duration"
    , "Customer Feedback Number", "Gender Number",
    "Marital Status Number", "Location Number", "Policy Type Number",
    "Education Level Number", "Property Type Number"
]
X = train1_df[feature].values
Y = train1_df[target].values

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
test1_train1_df = pd.read_csv("test.csv")
test1_train1_df.isnull().sum().sort_values(ascending=False)
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
test1_train1_df.head()

test1_data = test1_train1_df.values
X_test1 = test1_data[:, 0:]
Y_test1 = model.predict(X_test1, verbose=0)

submission1_train1_df = pd.DataFrame({"id": np.arange(1200000, 1200000 + len(Y_test1)), "Premium Amount": Y_test1.flatten()})
submission1_train1_df.to_csv("bfkeras_submission1.csv", index=False)
