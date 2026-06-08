# ============================================================
# 신규 SBR 시설 LCC 대시보드 V7.3 - Streamlit 이식본
# 원 Colab ipywidgets 대시보드의 계산 구조와 화면 구성을 최대한 동일하게 변환
# ============================================================

from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# [0] Streamlit 기본 설정
# ============================================================
st.set_page_config(
    page_title="신규 SBR 시설 대상 AI-LCC 시뮬레이터 대시보드(V7.3)",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# ============================================================
# [1] 한글 폰트 설정
# - Windows 로컬: C:/Windows/Fonts/malgun.ttf 직접 등록
# - Streamlit Cloud/Linux: NanumGothic이 설치되어 있으면 자동 사용
# ============================================================
def set_korean_font():
    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            mpl.rc("font", family=font_name)
            plt.rcParams["font.family"] = font_name
            mpl.rcParams["axes.unicode_minus"] = False
            return font_name

    # fallback: 설치된 폰트 이름으로 한 번 더 시도
    for font_name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        try:
            mpl.rc("font", family=font_name)
            plt.rcParams["font.family"] = font_name
            mpl.rcParams["axes.unicode_minus"] = False
            return font_name
        except Exception:
            continue
    return None

KOREAN_FONT = set_korean_font()

# ============================================================
# [2] CAPEX 회귀계수 및 범용 ML 모델 로드
# ============================================================
# 전국 SBR 단독공정(메인)
# N=119, R2=0.6357, p=2.04e-27
A_COEF = 18_056_027.0
B_COEF = 0.8723

UNIVERSAL_FEATURES = [
    "inflow_m3", "water_temp", "in_pH", "in_BOD",
    "in_TN", "in_SS", "in_TP", "doy_sin", "doy_cos"
]

@st.cache_resource
def load_models():
    """Colab outputs에서 내려받은 범용 ML 모델 3종을 상대경로로 로드."""
    ml_chem_model = joblib.load(MODEL_DIR / "rf_chem_cost_won_universal.pkl")
    ml_sludge_model = joblib.load(MODEL_DIR / "rf_sludge_cost_won_universal.pkl")
    ml_elec_model = joblib.load(MODEL_DIR / "rf_elec_won_universal.pkl")
    return ml_chem_model, ml_sludge_model, ml_elec_model

model_elec, model_chem, model_sludge = load_models()

# Colab V7.3 코드와 동일한 변수명으로 연결
ml_elec_model = model_elec
ml_chem_model = model_chem
ml_sludge_model = model_sludge
# ============================================================
# [3] 단가 / 증액률 상수
# ============================================================
sludge_unit_price = 130_000   # 원/톤
capex_ai_inc = 0.088          # AI 도입 CAPEX 증액률 8.8%
ELEC_WON_PER_M3_GLOBAL = 700.0

# ============================================================
# [4] 이론 비용 계산 함수 - Colab V7.3과 동일 구조
# ============================================================
def calculate_theoretical_day_cost(Q, BOD, SS, TP):
    """
    AI 시나리오 일별 이론 비용, 벡터 연산.
    · 슬러지: 생물학적 발생량 × 탈수 기준 × 처리 단가
    · 약품  : Al/P 몰비 이론식, M_ratio=1.0, PAC 385원/kg
    """
    Y_coef, b_fss, moisture = 0.4, 0.3, 0.80
    s_bio_kgd = Q * (Y_coef * BOD + b_fss * SS) / 1000
    sludge_ton_day = s_bio_kgd / (1 - moisture) / 1000
    daily_sludge_cost = sludge_ton_day * sludge_unit_price

    MW_Al, MW_P, M_ratio, PAC_price = 26.98, 30.97, 1.0, 385.0
    tp_removed = np.maximum(TP - 0.2, 0)
    theoretical_chem_kg = Q * tp_removed * (MW_Al / MW_P) * M_ratio / 1000
    daily_chem_cost = theoretical_chem_kg * PAC_price

    return daily_sludge_cost, daily_chem_cost

# ============================================================
# [5] 합성 환경 데이터 생성 함수 - Colab V7.3과 동일 구조
# ============================================================
def generate_synthetic_environmental_days(base_Q, base_BOD, base_SS,
                                          base_TN, base_TP, base_Temp,
                                          trend_rate, noise_level):
    """20년치(7,300일) 합성 환경 데이터 생성."""
    days = 7300
    time_idx = np.arange(days)
    years = time_idx / 365.0
    trend_factor = (1 + trend_rate) ** years
    season_sin = np.sin(2 * np.pi * time_idx / 365)
    season_cos = np.cos(2 * np.pi * time_idx / 365)

    rng = np.random.default_rng()

    df_syn = pd.DataFrame(index=time_idx)
    df_syn["inflow_m3"] = np.maximum(
        base_Q * trend_factor + rng.normal(0, base_Q * noise_level, days),
        base_Q * 0.5
    )
    df_syn["in_BOD"] = np.maximum(
        base_BOD * trend_factor - 15 * season_sin
        + rng.normal(0, base_BOD * noise_level, days),
        10.0
    )
    df_syn["in_SS"] = np.maximum(
        base_SS * trend_factor + rng.normal(0, base_SS * noise_level, days),
        10.0
    )
    df_syn["in_TN"] = np.maximum(
        base_TN * trend_factor + rng.normal(0, base_TN * noise_level, days),
        5.0
    )
    df_syn["in_TP"] = np.maximum(
        base_TP * trend_factor + rng.normal(0, base_TP * noise_level, days),
        0.3
    )
    df_syn["water_temp"] = np.maximum(base_Temp + 8 * season_sin, 4.0)
    df_syn["in_pH"] = np.clip(
        7.2 + 0.2 * season_sin + rng.normal(0, 0.1, days),
        6.0, 8.5
    )
    df_syn["doy_sin"] = season_sin
    df_syn["doy_cos"] = season_cos

    return df_syn

# ============================================================
# [6] 시각화 함수 - Colab V7.3과 동일한 막대그래프 구조
# ============================================================
def visualize_lcc_single_plot_v7(years, total_base, total_ai):
    plt.close("all")
    sns.set_style("whitegrid")
    if KOREAN_FONT:
        mpl.rc("font", family=KOREAN_FONT)
        plt.rcParams["font.family"] = KOREAN_FONT
    mpl.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(18, 9.5))
    width = 0.35

    ax.bar(
        years - width / 2,
        total_base,
        width,
        label="기존 운영",
        color="#475569",
        alpha=0.9,
        edgecolor="white",
        zorder=3,
    )
    ax.bar(
        years + width / 2,
        total_ai,
        width,
        label="AI 운영",
        color="#0D9488",
        alpha=0.9,
        edgecolor="white",
        zorder=3,
    )

    for idx in [0, 20]:
        ax.text(
            idx - width / 2,
            total_base[idx] * 1.01,
            f" {total_base[idx]:.1f}억 ",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="#334155", edgecolor="none", boxstyle="round,pad=0.4", alpha=0.95),
            zorder=4,
        )
        ax.text(
            idx + width / 2,
            total_ai[idx] * 1.01,
            f" {total_ai[idx]:.1f}억 ",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="#0F766E", edgecolor="none", boxstyle="round,pad=0.4", alpha=0.95),
            zorder=4,
        )

    ax.set_title("생애주기비용(LCC) 20년 누적 막대그래프", fontsize=26, fontweight="bold", pad=25)
    ax.set_xlabel("누적 연차(년)", fontsize=18, fontweight="bold", labelpad=15)
    ax.set_ylabel("LCC 누적 비용 (억원) [초기 투자비 + 누적 운영비]", fontsize=18, fontweight="bold", labelpad=15)
    ax.set_xticks(years)
    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")
    ax.set_xlim(-1, 21)

    max_y = max(total_base[-1], total_ai[-1])
    min_y = min(total_base[0], total_ai[0])
    ax.set_ylim(bottom=min_y * 0.95, top=max_y * 1.15)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y", zorder=0)

    total_saving = total_base[-1] - total_ai[-1]
    if total_saving >= 0:
        saving_color = "#1E40AF"
        saving_bgcolor = "#DBEAFE"
        saving_border = "#3B82F6"
        saving_label = f"20년 총 LCC 절감액: {total_saving:.1f} 억원"
    else:
        saving_color = "#991B1B"
        saving_bgcolor = "#FEE2E2"
        saving_border = "#EF4444"
        saving_label = f"20년 총 LCC 추가 비용(AI 불리): {abs(total_saving):.1f} 억원"

    box_y_pos = 0.96
    ax.text(
        0.85,
        box_y_pos,
        saving_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=saving_color,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=saving_bgcolor, edgecolor=saving_border, linewidth=2.0),
        zorder=5,
    )
    ax.legend(
        prop={"size": 20, "weight": "bold"},
        loc="upper left",
        bbox_to_anchor=(0.08, 0.03 + box_y_pos),
        ncol=2,
        framealpha=0.95,
        edgecolor="gray",
        shadow=False,
    )

    plt.tight_layout()
    return fig

# ============================================================
# [7] UI - Colab ipywidgets 레이아웃을 Streamlit form으로 변환
# ============================================================
st.markdown("<h2 style='color:#0F766E;'>신규 SBR 시설 대상 AI-LCC 시뮬레이터 대시보드(V7.3)</h2>", unsafe_allow_html=True)

with st.form("lcc_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**[1] 신규 시설 초기 조건**")
        w_capacity = st.number_input("시설용량(m³/일):", min_value=1, max_value=1_000_000, value=6200, step=100)
        w_Q = st.number_input("가동 유량(m³/일):", min_value=1, max_value=1_000_000, value=5617, step=100)
        w_bod = st.number_input("유입 BOD(mg/L):", min_value=0.0, max_value=1000.0, value=247.1, step=0.1)
        w_ss = st.number_input("유입 SS(mg/L):", min_value=0.0, max_value=1000.0, value=269.1, step=0.1)
        w_tp = st.number_input("유입 T-P(mg/L):", min_value=0.0, max_value=100.0, value=6.88, step=0.01)
        w_tn = st.number_input("유입 T-N(mg/L):", min_value=0.0, max_value=500.0, value=40.0, step=0.1)
        w_temp = st.number_input("유입 수온(℃):", min_value=-10.0, max_value=50.0, value=15.0, step=0.1)

    with col_right:
        st.markdown("**[2] 시나리오 변수**")
        w_trend = st.slider("연간 비용상승률(%):", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        w_noise = st.slider("불확실성 노이즈(%):", min_value=0.0, max_value=50.0, value=30.0, step=1.0)
        w_sim_cnt = st.selectbox("몬테카를로 반복(회):", options=[10, 30, 50, 100, 300, 500, 1000], index=0)
        w_discount = st.slider("사회적 할인율(%):", min_value=1.0, max_value=6.0, value=4.5, step=0.1)
        st.markdown(
            "<p style='color:#64748B; font-size:12px; margin:8px 0 2px;'><b>[3] AI OPEX 가정</b></p>",
            unsafe_allow_html=True,
        )
        w_elec_save = st.slider("AI 전력 절감률(%):", min_value=0.0, max_value=30.0, value=10.0, step=0.5)

    submitted = st.form_submit_button("▶ 몬테카를로 LCC 통합 시뮬레이션 실행", use_container_width=True)

st.divider()

# ============================================================
# [8] Streamlit 시뮬레이션 엔진
# ============================================================
def run_integrated_lcc_streamlit(CAPACITY, Q, BOD, SS, TP, TN, TEMP,
                                 trend_pct, noise_pct, n_sim, discount_pct, elec_save_pct):
    trend = trend_pct / 100.0
    noise = noise_pct / 100.0
    r_disc = discount_pct / 100.0
    elec_save = elec_save_pct / 100.0

    capex_base = A_COEF * (CAPACITY ** B_COEF)
    capex_ai = capex_base * (1 + capex_ai_inc)

    annual_base_matrix = np.zeros((n_sim, 20))
    annual_ai_matrix = np.zeros((n_sim, 20))

    progress = st.progress(0, text="시뮬레이션 준비 중")

    for sim in range(n_sim):
        df_day = generate_synthetic_environmental_days(Q, BOD, SS, TN, TP, TEMP, trend, noise)
        df_model = df_day[UNIVERSAL_FEATURES].copy()

        daily_sludge_theory, daily_chem_theory = calculate_theoretical_day_cost(
            df_day["inflow_m3"].values,
            df_day["in_BOD"].values,
            df_day["in_SS"].values,
            df_day["in_TP"].values,
        )

        if ml_sludge_model is not None:
            daily_sludge_base = np.maximum(ml_sludge_model.predict(df_model), 0)
        else:
            daily_sludge_base = daily_sludge_theory * 1.5

        if ml_chem_model is not None:
            daily_chem_base = np.maximum(ml_chem_model.predict(df_model), 0)
        else:
            daily_chem_base = daily_chem_theory * 2.0

        if ml_elec_model is not None:
            daily_elec_base = np.maximum(ml_elec_model.predict(df_model), 0)
        else:
            daily_elec_base = df_day["inflow_m3"].values * ELEC_WON_PER_M3_GLOBAL

        daily_sludge_ai = daily_sludge_theory
        daily_chem_ai = daily_chem_theory
        daily_elec_ai = daily_elec_base * (1 - elec_save)

        daily_total_base = daily_sludge_base + daily_chem_base + daily_elec_base
        daily_total_ai = daily_sludge_ai + daily_chem_ai + daily_elec_ai

        for y in range(20):
            disc = 1 / ((1 + r_disc) ** (y + 1))
            annual_base_matrix[sim, y] = np.sum(daily_total_base[y * 365:(y + 1) * 365]) * disc
            annual_ai_matrix[sim, y] = np.sum(daily_total_ai[y * 365:(y + 1) * 365]) * disc

        progress.progress((sim + 1) / n_sim, text=f"시뮬레이션 중: {sim + 1}/{n_sim}")

    progress.empty()

    cumsum_base = np.zeros((n_sim, 21))
    cumsum_ai = np.zeros((n_sim, 21))

    for sim in range(n_sim):
        cumsum_base[sim, 0] = capex_base
        cumsum_ai[sim, 0] = capex_ai
        for y in range(20):
            cumsum_base[sim, y + 1] = cumsum_base[sim, y] + annual_base_matrix[sim, y]
            cumsum_ai[sim, y + 1] = cumsum_ai[sim, y] + annual_ai_matrix[sim, y]

    cumsum_base_억 = cumsum_base / 1e8
    cumsum_ai_억 = cumsum_ai / 1e8

    total_base_mean = np.mean(cumsum_base_억, axis=0)
    total_ai_mean = np.mean(cumsum_ai_억, axis=0)
    arr_base_final = cumsum_base_억[:, -1]
    arr_ai_final = cumsum_ai_억[:, -1]
    mean_saving = np.mean(arr_base_final) - np.mean(arr_ai_final)

    mean_base = np.mean(arr_base_final)
    saving_pct = (mean_saving / mean_base * 100) if mean_base != 0 else float("nan")

    return total_base_mean, total_ai_mean, arr_base_final, arr_ai_final, mean_saving, saving_pct

if submitted:
    with st.spinner("몬테카를로 LCC 통합 시뮬레이션 실행 중입니다."):
        total_base_mean, total_ai_mean, arr_base_final, arr_ai_final, mean_saving, saving_pct = run_integrated_lcc_streamlit(
            w_capacity, w_Q, w_bod, w_ss, w_tp, w_tn, w_temp,
            w_trend, w_noise, int(w_sim_cnt), w_discount, w_elec_save
        )

    if mean_saving >= 0:
        saving_color = "#1D4ED8"
        saving_sign = ""
        saving_verdict = "예측 절감액"
    else:
        saving_color = "#B91C1C"
        saving_sign = "▲"
        saving_verdict = "비용 증가 (AI 불리)"

    html_report = f"""
    <div style='background:#FAFBFD; padding:25px; border-radius:12px;
                margin-bottom:25px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
                font-family: NanumGothic, sans-serif;'>

      <div style='display:flex; justify-content:space-around; align-items:center;'>

        <div style='text-align:center;'>
          <span style='font-size:13px; color:#64748B; font-weight:bold;
                       display:block; margin-bottom:2px;'>
          </span>
          <span style='font-size:16px; color:#1E293B; font-weight:bold; display:block; margin-bottom:8px;'>
            기존 운영 누적 총 비용 (20년)
          </span>
          <span style='font-size:36px; color:#2C3E50; font-weight:800; letter-spacing:-1px;'>
            {np.mean(arr_base_final):.1f} 억원
          </span>
          <span style='font-size:14px; color:#64748B; font-weight:bold; display:block; margin-top:6px;'>
            ({np.percentile(arr_base_final, 5):.1f}~{np.percentile(arr_base_final, 95):.1f}억)
          </span>
        </div>

        <div style='text-align:center;'>
          <span style='font-size:16px; color:#0F766E; font-weight:bold; display:block; margin-bottom:8px;'>
            AI 최적화 누적 총 비용 (20년)
          </span>
          <span style='font-size:36px; color:#0D9488; font-weight:800; letter-spacing:-1px;'>
            {np.mean(arr_ai_final):.1f} 억원
          </span>
          <span style='font-size:14px; color:#64748B; font-weight:bold; display:block; margin-top:6px;'>
            ({np.percentile(arr_ai_final, 5):.1f}~{np.percentile(arr_ai_final, 95):.1f}억)
          </span>
        </div>

        <div style='text-align:center;'>
          <span style='font-size:14px; color:{saving_color}; font-weight:bold;
                       display:block; margin-bottom:6px;'>{saving_verdict}</span>
          <span style='font-size:38px; color:{saving_color}; font-weight:800; letter-spacing:-1px;'>
            {saving_sign}{mean_saving:.1f} 억원
          </span>
        </div>

      </div>
    </div>
    """
    # HTML 요약 박스는 st.markdown보다 components.html이 안정적입니다.
    # Streamlit이 HTML을 코드처럼 표시하는 문제를 방지합니다.
    components.html(html_report, height=185, scrolling=False)

    years_arr = np.arange(21)
    fig = visualize_lcc_single_plot_v7(years_arr, total_base_mean, total_ai_mean)
    st.pyplot(fig, clear_figure=True)
else:
    st.info("입력값을 확인한 뒤 '▶ 몬테카를로 LCC 통합 시뮬레이션 실행' 버튼을 누르세요.")
