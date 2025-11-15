import plot_drawer

import numpy as np

# 오차율(%) 계산 함수
def error_percentage(P_real, P_pred):
    return np.abs((P_real - P_pred) / P_real) * 100

def calculate_error():
    # 실제 가격
    P_actual = plot_drawer.plot_actual()

    # 모델 예측값
    P_model  = plot_drawer.plot_model()
    _, P_reg = plot_drawer.plot_linear_regressive()
    
    # 각 시점별 오차율
    err_model = error_percentage(P_actual, P_model)
    err_reg   = error_percentage(P_actual, P_reg)

    # 평균 오차율(MAPE)
    mape_model = np.mean(err_model)
    mape_reg   = np.mean(err_reg)

    print(f"\n단순 수치 모델 MAPE: {mape_model:.2f}%")
    print(f"선형 회귀 모델 MAPE: {mape_reg:.2f}%")

    return mape_model, mape_reg, err_model, err_reg
