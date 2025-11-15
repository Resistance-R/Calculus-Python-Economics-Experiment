import data_reader
import plot_drawer
import error_rate

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = data_reader.import_data()
mape_model, mape_reg, err_model, err_reg = error_rate.calculate_error()

if __name__ == "__main__":
    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3)

    """
    -----------------------------
     (1) 왼쪽: 이변수 함수 곡면 그래프
    -----------------------------
    """
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    T, VV, P_surf = plot_drawer.surface_data()

    surf = ax1.plot_surface(T, VV, P_surf,
                            cmap='viridis',
                            alpha=0.8)

    ax1.set_xlabel("Time: t (index)")
    ax1.set_ylabel("Volume: V")
    ax1.set_zlabel("Price: P")
    ax1.set_title("Surface of a two-variable function P(t, V)")

    # 실제 데이터 점도 찍기
    t_vals, V_vals, P_real = plot_drawer.actual_points_3d()
    ax1.scatter(t_vals, V_vals, P_real,
                color="red", s=10, label="Actual Data")
    ax1.legend()

    """
    -----------------------------
     (2) 오른쪽: 기존 2D 시계열 그래프
    -----------------------------
    """
    reg, P_reg = plot_drawer.plot_linear_regressive()
    ax2 = fig.add_subplot(gs[0, 2])

    dP_dt, dP_dV = plot_drawer.partial_diff()

    ax2.plot(df["날짜"], plot_drawer.plot_actual(),
             label="Actual Price: P_real", linewidth=2)
    ax2.plot(df["날짜"], plot_drawer.plot_model(),
             label="Numerical Model: P_model", linestyle="--")
    ax2.plot(df["날짜"], P_reg,
             label="Linear Regression Prediction: P_reg", linestyle=":")
    ax2.plot(df["날짜"], dP_dt,
             label="Price changes over time: dP/dt", linestyle="-.")
    ax2.plot(df["날짜"], dP_dV,
             label="Price changes based on trading volume: dP/dV", linestyle="-")

    ax2.set_xlabel("date")
    ax2.set_ylabel("Price (USD)")
    ax2.set_title("Actual vs Model vs Regression (Time-Price)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 회귀 계수 출력
    print("\n선형회귀 계수:")
    print(f"    Intercept = {reg.intercept_:.4f}")
    print(f"    a (t 계수) = {reg.coef_[0]:.6f}")
    print(f"    b (V 계수) = {reg.coef_[1]:.10f}")
    print(f"    c (sin 계수) = {reg.coef_[2]:.6f}")

    """
    -----------------------------
     (3) 중간 - 벡터장: 
     수치 모델로 분석한 시장의 추세
    -----------------------------
    """
    ax4 = fig.add_subplot(gs[0, 1])

    T, VV, dP_T, dP_V = plot_drawer.gradient_field()

    ax4.quiver(T, VV, dP_T, dP_V, angles='xy', scale_units='xy', scale=1)
    ax4.set_xlabel("Time: t (index)")
    ax4.set_ylabel("Volume: V")
    ax4.set_title("Gradient Field ∇P(t,V) — market trend direction")

    """
    -----------------------------
     (4) 오차율 표시
    -----------------------------
    """
    ymax = max(max(err_model), max(err_reg))

    ax3 = fig.add_subplot(gs[1, :])
    ax3.text(df["날짜"].iloc[-1], ymax * 0.95,
         f"MAPE(model) = {mape_model:.2f}%",
         horizontalalignment='right',
         fontsize=10, color="blue")

    ax3.text(df["날짜"].iloc[-1], ymax * 0.05,
            f"MAPE(reg) = {mape_reg:.2f}%",
            horizontalalignment='right',
            fontsize=10, color="orange")
    
    ax3.plot(df["날짜"], err_model, label="Model Error (%)", linestyle="--")
    ax3.plot(df["날짜"], err_reg, label="Regression Error (%)", linestyle=":")

    ax3.set_ylabel("Error Rate(%)")
    ax3.set_xlabel("Date")
    ax3.set_title("Error rate by model and time point(%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()