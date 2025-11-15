import data_reader

import numpy as np
from sklearn.linear_model import LinearRegression

df = data_reader.import_data()
t = df["t"].values
V = df["거래량"].values


""" 모델 정의와 편미분 """
def model_function(t_val, V_val):
    # 예측 모델 정의
    return 0.5 * t_val + 0.01 * V_val + 10 * np.sin(0.5 * t_val) # P(t, V) = 0.5t + 0.01V + 10sin(0.5t)

def dP_dt(t_val, V_val):
    # 시간에 대한 편미분 ∂P/∂t = 0.5 + 5 cos(0.5 t)
    return 0.5 + 5 * np.cos(0.5 * t_val)

def dP_dV(t_val, V_val):
    # 거래량에 대한 편미분 ∂P/∂V = 0.001
    return 0.01 * np.ones_like(t_val, dtype=float)

""" 플롯에 기록할 값 구하기 """
def plot_actual():
    P_real = df["종가"].values # 실제 데이터 저장
    return P_real # 실제 데이터 반환

def plot_model():
    return model_function(df["t"].values, df["거래량"].values)

def plot_linear_regressive():
    # 사인 항 포함
    sin_term = np.sin(0.5 * t)

    # 설계행렬 X = [t, V, sin(0.5t)]
    X = np.column_stack([t, V, sin_term])

    reg = LinearRegression()
    reg.fit(X, plot_actual())

    # 회귀 예측
    P_reg = reg.predict(X)

    return reg, P_reg

def surface_data(num_t=50, num_V=50):
    """
    이변수 함수 P(t,V)의 곡면을 그리기 위한 격자 데이터 생성
    반환: T(시간 격자), VV(거래량 격자), P_surf(모델 값)
    """
    t_min, t_max = df["t"].min(), df["t"].max()
    V_min, V_max = df["거래량"].min(), df["거래량"].max()

    t_vals = np.linspace(t_min, t_max, num_t)
    V_vals = np.linspace(V_min, V_max, num_V)

    T, VV = np.meshgrid(t_vals, V_vals)
    P_surf = model_function(T, VV)

    return T, VV, P_surf

def actual_points_3d():
    """
    곡면 위에 찍을 실제 데이터의 (t, V, P_real) 좌표 반환
    """
    P_real = df["종가"].values
    return t, V, P_real

def partial_diff():
    """
    모델 기반 편미분
    """
    dP_dt_now = dP_dt(df["t"].values, df["거래량"].values)
    dP_dV_now = dP_dV(df["t"].values, df["거래량"].values)
    return dP_dt_now, dP_dV_now

def gradient_field():
    """
    수치 모델 기반 벡터장 구하기
    """
    T, VV, _ = surface_data()  # T: 시간 격자, VV: 거래량 격자
    dP_T = dP_dt(T, VV)
    dP_V = dP_dV(T, VV)
    return T, VV, dP_T, dP_V