import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, render_template

# 1. 初始化 Flask 網站應用程式
app = Flask(__name__)

# 2. 載入 Keras 模型與 Scaler 預處理器
print("--------------------------------------------------")
print("系統啟動中...")
model = None
scaler = None

try:
    print("正在載入 Keras 模型 (my_fintech_model.h5)...")
    
    # --- 關鍵修正 ---
    # 加入 compile=False
    # 原因：我們只需要用模型來「預測」，不需要載入「訓練」設定 (如優化器、MAE、RMSE)。
    # 這樣可以完全避開 'Could not deserialize keras.metrics.mae' 的錯誤。
    model = tf.keras.models.load_model('my_fintech_model.h5', compile=False)
    
    print("正在載入預處理器 (my_fintech_scaler.pkl)...")
    scaler = joblib.load('my_fintech_scaler.pkl')
    
    print("✅ 載入成功！保險預測網站準備就緒。")
except Exception as e:
    print("❌ 嚴重錯誤：載入失敗！")
    print(f"錯誤訊息: {e}")
    print("請確認檔案 'my_fintech_model.h5' 和 'my_fintech_scaler.pkl' 是否在資料夾中。")
print("--------------------------------------------------")

# --- 路由設定 ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 確保模型與 Scaler 已載入
    if not model or not scaler:
        return render_template('index.html', prediction_text="系統錯誤：模型未正確載入，請聯繫管理員。")

    try:
        # A. 接收網頁表單傳來的 14 個欄位資料
        # 注意：這裡的欄位名稱必須跟 HTML 的 name="xxx" 一模一樣
        age = float(request.form['Age'])
        annual_income = float(request.form['Annual_Income'])
        dependents = int(request.form['Number_of_Dependents'])
        health_score = float(request.form['Health_Score'])
        vehicle_age = float(request.form['Vehicle_Age'])
        credit_score = float(request.form['Credit_Score'])
        insurance_duration = float(request.form['Insurance_Duration'])
        
        # 類別資料 (HTML 下拉選單傳回的是字串 "0", "1" 等，需轉為 int)
        customer_feedback = int(request.form['Customer_Feedback'])
        gender = int(request.form['Gender'])
        marital_status = int(request.form['Marital_Status'])
        location = int(request.form['Location'])
        policy_type = int(request.form['Policy_Type'])
        education_level = int(request.form['Education_Level'])
        property_type = int(request.form['Property_Type'])

        # B. 資料預處理 (Scaler)
        # 根據 2ndopt_keras.py，只有這三個欄位需要標準化
        # 建立臨時 numpy array 進行轉換
        features_to_scale = np.array([[annual_income, health_score, credit_score]])
        
        # 呼叫 Scaler 進行轉換
        scaled_values = scaler.transform(features_to_scale)
        
        # 取出轉換後的值
        s_income = scaled_values[0][0]
        s_health = scaled_values[0][1]
        s_credit = scaled_values[0][2]

        # C. 組合最終輸入向量
        # ⚠️ 順序必須跟訓練程式 (2ndopt_keras.py) 的 feature 列表完全一致
        final_input = np.array([[
            age, 
            s_income,       # Scaled
            dependents, 
            s_health,       # Scaled
            vehicle_age, 
            s_credit,       # Scaled
            insurance_duration, 
            customer_feedback, 
            gender, 
            marital_status, 
            location, 
            policy_type, 
            education_level, 
            property_type
        ]])

        # D. 進行預測
        prediction = model.predict(final_input)
        result_value = prediction[0][0] # 取得預測金額

        # 格式化金額 (加逗號與兩位小數)
        result_text = f"${result_value:,.2f}"
        
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        error_msg = f"發生錯誤: {str(e)}"
        print(error_msg)
        return render_template('index.html', prediction_text=error_msg)

if __name__ == "__main__":
    app.run(debug=True, port=5000)