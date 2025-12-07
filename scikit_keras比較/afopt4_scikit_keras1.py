import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("train_preprocessed_.csv")
#df_train.info() #檢查資料用


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, r2_score # 引入迴歸指標
import joblib #匯出模型


def insurance_model(model, data, predictors, outcome, t_size, name): #定義函式，輸入模型、資料、用於預測的變數、結果變數、測試資料比例
    X = data[predictors]
    y = data[outcome]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t_size, random_state= 100 #指定亂數種子為100
    )

    X_train = X_train[predictors]
    X_test = X_test[predictors]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2_result = r2_score(y_test, predictions)
    mse_result = mean_squared_error(y_test, predictions)
    rmse_result = np.sqrt(mse_result)
    joblib.dump(model,f"Insurance-{name}-mode.afopt4.pkl", compress=3) #匯出模型，檔名包含模型名稱
    return r2_result, mse_result, rmse_result #回傳R平方、均方誤差、均方根誤差


# 1 線性回歸
outcome_var = "Premium Amount"
model_lr =  LinearRegression()
predictor_var = [
    "Age", "Annual Income", "Number of Dependents", "Health Score", "Vehicle Age", "Credit Score", "Insurance Duration"
    , "Customer Feedback Number", "Gender Number",
    "Marital Status Number", "Location Number", "Policy Type Number",
    "Education Level Number", "Property Type Number"
]
r2, mse, rmse = insurance_model(
    model=model_lr,
    data=df_train,
    predictors=predictor_var,
    outcome=outcome_var,
    t_size=0.3,
    name = "Linear Regression",
)
print(f"Linear Regression")
print(f"R^2 score:{r2}")
print(f"MSE:{mse}")
print(f"RMSE:{rmse}")


# 2 隨機森林
outcome_var = "Premium Amount"
model_rf =  RandomForestRegressor(    
    n_estimators=50, # 樹的數量
    max_depth=15, #限制最大深度為 15，以防止過度擬合
    n_jobs=4,      # 使用的CPU核心數，加速訓練
    verbose=2,      # 印出每棵樹的訓練進度
    random_state=100)
predictor_var = ["Age", "Annual Income", "Number of Dependents",
    "Health Score", "Vehicle Age", "Credit Score",
    "Insurance Duration", "Customer Feedback Number",
    "Gender Number", "Marital Status Number",
    "Location Number", "Policy Type Number",
    "Education Level Number", "Property Type Number"
]
r2, mse, rmse = insurance_model(
    model=model_rf,
    data=df_train,
    predictors=predictor_var,
    outcome=outcome_var,
    t_size=0.3,
    name= "RF Regressor",
)
print(f"Random Forest Regressor")
print(f"R^2 score:{r2}")
print(f"MSE:{mse}")
print(f"RMSE:{rmse}")

