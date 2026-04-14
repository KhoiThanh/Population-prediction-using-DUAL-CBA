"""
Demo Application: Vietnam Population Forecast 2025-2034
Hiển thị dữ liệu lịch sử và dự báo dân số cho 63 tỉnh/thành phố Việt Nam
Bao gồm XAI LIME để giải thích dự báo
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import joblib

try:
    from xai_lime import explain_area_window, lag_feature_names_for_window_1996_2006
    from model import BahdanauAttention
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False

# ============================================================================
# CONFIG & LOAD DATA
# ============================================================================

st.set_page_config(
    page_title="Dự báo Dân số Việt Nam",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Ứng dụng Dự báo Dân số Việt Nam (2025-2034)")
st.markdown("Dữ liệu từ 63 tỉnh/thành phố | Theo dõi xu hướng dân số trong 10 năm tới | XAI-LIME Explainability")

ART_DIR = Path("artifacts")

# Load data
@st.cache_data
def load_data():
    # Historical data (2011-2024)
    df_hist = pd.read_csv("Vietnam_Final_63_Provinces.csv")

    # Forecast data (2025-2034)
    df_forecast = pd.read_csv("artifacts/vietnam_transfer_forecast_10y.csv")

    return df_hist, df_forecast

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """Load trained model and scaler for XAI"""
    if not XAI_AVAILABLE:
        return None, None

    model_path = ART_DIR / "dual_cba_best.keras"
    scaler_path = ART_DIR / "global_minmax_scaler.joblib"

    model = None
    scaler = None

    # Try to load pre-trained model
    if model_path.exists():
        try:
            model = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"BahdanauAttention": BahdanauAttention},
            )
            model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
        except Exception as e:
            st.warning(f"Không thể load model: {e}")
            model = None

    # Try to load scaler
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Không thể load scaler: {e}")
            scaler = None

    return model, scaler

@st.cache_data
def build_lime_training_data_vietnam(df_hist, _scaler):
    """Build LIME training data from all 63 provinces using 2014-2024 window"""
    if _scaler is None:
        return None

    # Years 2011-2024, indices 3-13 correspond to 2014-2024 (11 years window)
    year_cols = [col for col in df_hist.columns if col.endswith('_VN') and col[:-3].isdigit()]
    year_cols_sorted = sorted(year_cols, key=lambda x: int(x[:-3]))

    # Get indices for 2014-2024 (11 years window)
    years = [int(col[:-3]) for col in year_cols_sorted]
    start_idx = years.index(2014) if 2014 in years else 0
    end_idx = start_idx + 11

    window_cols = year_cols_sorted[start_idx:end_idx]

    X_rows = []
    for _, row in df_hist.iterrows():
        try:
            y_raw = np.array([float(row[col]) for col in window_cols], dtype=np.float32)
            if not np.isfinite(y_raw).all():
                continue
            y_scaled = _scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
            if len(y_scaled) == 11:
                X_rows.append(y_scaled)
        except Exception:
            continue

    return np.array(X_rows, dtype=np.float32) if X_rows else None

try:
    df_hist, df_forecast = load_data()
except FileNotFoundError as e:
    st.error(f"Lỗi: Không tìm thấy file dữ liệu - {e}")
    st.stop()

# Load model and scaler
model, scaler = load_model_and_scaler()
lime_training_data = build_lime_training_data_vietnam(df_hist, scaler) if scaler is not None else None

# ============================================================================
# SIDEBAR - SELECT PROVINCE
# ============================================================================

with st.sidebar:
    st.header("⚙️ Cài đặt")

    # Get list of areas from historical data
    areas = sorted(df_hist["Area_Name"].unique())
    selected_area = st.selectbox(
        "Chọn tỉnh/thành phố:",
        options=areas,
        index=0
    )

    st.markdown("---")
    st.markdown("**Thông tin ứng dụng**")
    st.info(
        "- **Dữ liệu lịch sử**: 2011-2024\n"
        "- **Dự báo**: 2025-2034\n"
        "- **Số vùng**: 63 tỉnh/TP\n"
        "- Dữ liệu: Người (nghìn)"
    )

# ============================================================================
# FUNCTIONS FOR VISUALIZATION
# ============================================================================

def get_historical_data(area_name):
    """Lấy dữ liệu lịch sử cho một tỉnh"""
    row = df_hist[df_hist["Area_Name"] == area_name]
    if row.empty:
        return None, None

    row = row.iloc[0]

    # Extract year columns (format: YYYY_VN)
    year_cols = [col for col in df_hist.columns if col.endswith('_VN') and col[:-3].isdigit()]
    years_hist = sorted([int(col[:-3]) for col in year_cols])

    values = [float(row[f'{y}_VN']) if pd.notna(row[f'{y}_VN']) else None for y in years_hist]
    return years_hist, values

def get_historical_data_scaled(area_name):
    """Lấy dữ liệu lịch sử đã chuẩn hóa cho một tỉnh (dùng cho LIME)"""
    if scaler is None:
        return None, None

    row = df_hist[df_hist["Area_Name"] == area_name]
    if row.empty:
        return None, None

    row = row.iloc[0]

    # Extract year columns
    year_cols = [col for col in df_hist.columns if col.endswith('_VN') and col[:-3].isdigit()]
    years_hist = sorted([int(col[:-3]) for col in year_cols])

    # Get raw values
    y_raw = np.array([float(row[f'{y}_VN']) if pd.notna(row[f'{y}_VN']) else np.nan for y in years_hist], dtype=np.float32)

    # Scale using global scaler
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    return years_hist, y_scaled

def get_forecast_data(area_name):
    """Lấy dữ liệu dự báo cho một tỉnh"""
    row = df_forecast[df_forecast["Area_Name"] == area_name]
    if row.empty:
        return None, None

    row = row.iloc[0]

    # Extract prediction columns (format: pred_YYYY)
    pred_cols = [col for col in df_forecast.columns if col.startswith('pred_') and col[5:].isdigit()]
    years_forecast = sorted([int(col[5:]) for col in pred_cols])

    values = [float(row[f'pred_{y}']) if pd.notna(row[f'pred_{y}']) else None for y in years_forecast]
    return years_forecast, values

def get_window_2014_2024_scaled(area_name):
    """Lấy window 2014-2024 (11 years) đã chuẩn hóa cho một tỉnh (dùng cho LIME)"""
    if scaler is None:
        return None

    years_hist, y_scaled = get_historical_data_scaled(area_name)

    if years_hist is None or y_scaled is None:
        return None

    # Find indices for 2014-2024
    start_year = 2014
    end_year = 2024
    window_indices = [i for i, y in enumerate(years_hist) if start_year <= y <= end_year]

    if len(window_indices) == 11:  # Should be exactly 11 years
        return y_scaled[window_indices]
    else:
        return None

def explain_vietnam_area(area_name):
    """Giải thích dự báo cho một tỉnh bằng LIME"""
    if model is None or lime_training_data is None or scaler is None:
        return None

    window = get_window_2014_2024_scaled(area_name)
    if window is None:
        return None

    try:
        # Adjust LIME feature names for Vietnam window (2014-2024)
        years_window = list(range(2014, 2025))
        feature_names_vietnam = [f"lag_{i}_{y}" for i, y in enumerate(years_window, start=1)]

        from lime.lime_tabular import LimeTabularExplainer
        from xai_lime import lime_predict_fn_for_model

        explainer = LimeTabularExplainer(
            training_data=lime_training_data,
            feature_names=feature_names_vietnam,
            mode="regression",
            discretize_continuous=True,
            random_state=42,
        )

        instance = np.asarray(window, dtype=np.float32).reshape(-1)
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=lime_predict_fn_for_model(model),
            num_features=11,
        )

        # Local weights by feature index (robust against discretized label formatting)
        w_local = {
            feature_names_vietnam[idx]: float(w) for idx, w in exp.as_map().get(1, [])
        }

        df_local = (
            pd.DataFrame({"feature": list(w_local.keys()), "weight": list(w_local.values())})
            .assign(abs_weight=lambda d: d["weight"].abs())
            .sort_values("abs_weight", ascending=False)
            .drop(columns=["abs_weight"])
            .reset_index(drop=True)
        )

        # Global weights (sample from training data)
        rng = np.random.default_rng(42)
        n = len(lime_training_data)
        take = int(min(300, n))
        idx = rng.choice(n, size=take, replace=False) if take > 0 else np.array([], dtype=int)

        agg = {fn: [] for fn in feature_names_vietnam}
        for j in idx:
            inst = lime_training_data[j]
            exp_j = explainer.explain_instance(
                data_row=inst,
                predict_fn=lime_predict_fn_for_model(model),
                num_features=11,
            )
            wj = {
                feature_names_vietnam[idx]: float(w) for idx, w in exp_j.as_map().get(1, [])
            }
            for fn in feature_names_vietnam:
                agg[fn].append(abs(wj.get(fn, 0.0)))

        df_global = (
            pd.DataFrame({
                "feature": feature_names_vietnam,
                "mean_abs_weight": [float(np.mean(agg[fn])) if agg[fn] else 0.0 for fn in feature_names_vietnam],
            })
            .sort_values("mean_abs_weight", ascending=False)
            .reset_index(drop=True)
        )

        from dataclasses import dataclass

        @dataclass(frozen=True)
        class LimeResultVietnam:
            local_weights: pd.DataFrame
            global_abs_weights: pd.DataFrame

        return LimeResultVietnam(local_weights=df_local, global_abs_weights=df_global)

    except Exception as e:
        st.warning(f"Lỗi khi giải thích LIME: {e}")
        return None

def create_forecast_chart(area_name):
    """Tạo biểu đồ dự báo dân số"""
    years_hist, values_hist = get_historical_data(area_name)
    years_forecast, values_forecast = get_forecast_data(area_name)

    if not years_hist or not years_forecast:
        return None

    fig = go.Figure()

    # Dữ liệu lịch sử
    fig.add_trace(go.Scatter(
        x=years_hist,
        y=values_hist,
        mode='lines+markers',
        name='Dữ liệu lịch sử (2011-2024)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Dữ liệu dự báo
    fig.add_trace(go.Scatter(
        x=years_forecast,
        y=values_forecast,
        mode='lines+markers',
        name='Dự báo (2025-2034)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    # Thêm đường kết nối (năm 2024 -> 2025)
    last_hist = values_hist[-1]
    first_forecast = values_forecast[0]
    fig.add_trace(go.Scatter(
        x=[years_hist[-1], years_forecast[0]],
        y=[last_hist, first_forecast],
        mode='lines',
        name='Điểm kết nối',
        line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        title=f"Xu hướng dân số: {area_name}",
        xaxis_title="Năm",
        yaxis_title="Dân số (Nghìn người)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    fig.update_xaxes(dtick=2)

    return fig

def calculate_metrics(area_name):
    """Tính toán các chỉ số quan trọng"""
    years_hist, values_hist = get_historical_data(area_name)
    years_forecast, values_forecast = get_forecast_data(area_name)

    if not years_hist or not years_forecast:
        return None

    # Giá trị 2024 vs 2034
    pop_2024 = values_hist[-1]
    pop_2034 = values_forecast[-1]

    # Tính tốc độ tăng trưởng
    growth_10y_pct = ((pop_2034 - pop_2024) / pop_2024 * 100) if pop_2024 > 0 else 0
    avg_annual_growth = growth_10y_pct / 10

    # Dân số trung bình 2025-2034
    avg_forecast_pop = np.mean(values_forecast)

    return {
        "pop_2024": pop_2024,
        "pop_2034": pop_2034,
        "growth_10y_pct": growth_10y_pct,
        "avg_annual_growth": avg_annual_growth,
        "avg_forecast_pop": avg_forecast_pop,
        "change_absolute": pop_2034 - pop_2024
    }

def create_comparison_chart(areas):
    """Tạo biểu đồ so sánh tăng trưởng giữa các tỉnh"""
    data = []
    for area in areas:
        metrics = calculate_metrics(area)
        if metrics:
            data.append({
                "Area": area,
                "Growth_10y": metrics["growth_10y_pct"]
            })

    if not data:
        return None

    df_compare = pd.DataFrame(data).sort_values("Growth_10y", ascending=True)

    # Chọn top 15 tỉnh có tăng trưởng cao + 5 tỉnh giảm dân số nhiều nhất
    top_growth = df_compare.tail(15)

    fig = px.bar(
        top_growth,
        x="Growth_10y",
        y="Area",
        orientation="h",
        color="Growth_10y",
        color_continuous_scale="RdYlGn",
        title="Tốc độ tăng trưởng dân số 10 năm (2024-2034) - Top 15",
        labels={"Growth_10y": "Tăng trưởng (%)", "Area": "Tỉnh/Thành phố"},
        height=600
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        hovermode="closest"
    )

    return fig

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Metrics
metrics = calculate_metrics(selected_area)

if metrics:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dân số 2024",
            f"{metrics['pop_2024']:.1f}K",
            delta=None
        )

    with col2:
        st.metric(
            "Dân số dự báo 2034",
            f"{metrics['pop_2034']:.1f}K",
            delta=f"{metrics['change_absolute']:.1f}K"
        )

    with col3:
        st.metric(
            "Tăng trưởng 10 năm",
            f"{metrics['growth_10y_pct']:.2f}%",
            delta=None
        )

    with col4:
        st.metric(
            "Tăng trưởng trung bình/năm",
            f"{metrics['avg_annual_growth']:.2f}%",
            delta=None
        )

st.markdown("---")

# Chart
fig = create_forecast_chart(selected_area)
if fig:
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DETAILED DATA TABLE
# ============================================================================

st.subheader("📋 Dữ liệu chi tiết")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Dữ liệu lịch sử (2011-2024)**")
    years_hist, values_hist = get_historical_data(selected_area)
    if years_hist:
        df_table_hist = pd.DataFrame({
            "Năm": years_hist,
            "Dân số (Nghìn)": [f"{v:.1f}" if v else "N/A" for v in values_hist]
        })
        st.dataframe(df_table_hist, use_container_width=True, hide_index=True)

with col_right:
    st.markdown("**Dự báo (2025-2034)**")
    years_forecast, values_forecast = get_forecast_data(selected_area)
    if years_forecast:
        df_table_forecast = pd.DataFrame({
            "Năm": years_forecast,
            "Dân số dự báo (Nghìn)": [f"{v:.1f}" if v else "N/A" for v in values_forecast]
        })
        st.dataframe(df_table_forecast, use_container_width=True, hide_index=True)

# ============================================================================
# COMPARISON VIEW
# ============================================================================

st.markdown("---")
st.subheader("🔄 So sánh tỉnh/thành phố")

fig_compare = create_comparison_chart(areas)
if fig_compare:
    st.plotly_chart(fig_compare, use_container_width=True)

# ============================================================================
# INSIGHTS & ANALYSIS
# ============================================================================

if metrics:
    st.markdown("---")
    st.subheader("💡 Nhận xét & Phân tích")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        if metrics["growth_10y_pct"] > 15:
            st.success(f"""
            ✅ **Tăng trưởng cao**: {selected_area} có tốc độ tăng dân số **{metrics['growth_10y_pct']:.1f}%**
            trong 10 năm tới, tức khoảng **{metrics['avg_annual_growth']:.2f}%** mỗi năm.

            Cần chú ý đến nhu cầu cơ sở hạ tầng, giáo dục, và y tế.
            """)
        elif metrics["growth_10y_pct"] > 5:
            st.info(f"""
            ℹ️ **Tăng trưởng vừa phải**: {selected_area} có tốc độ tăng dân số **{metrics['growth_10y_pct']:.1f}%**
            trong 10 năm tới.
            """)
        elif metrics["growth_10y_pct"] > 0:
            st.warning(f"""
            ⚠️ **Tăng trưởng chậm**: {selected_area} có tốc độ tăng dân số tương đối thấp
            **{metrics['growth_10y_pct']:.1f}%** trong 10 năm tới.
            """)
        else:
            st.error(f"""
            🔴 **Giảm dân số**: {selected_area} dự báo sẽ giảm dân số **{abs(metrics['growth_10y_pct']):.1f}%**
            trong 10 năm tới.
            """)

    with col_info2:
        st.markdown(f"""
        **Số liệu chính:**
        - Dân số năm 2024: {metrics['pop_2024']:.1f}K người
        - Dân số năm 2034: {metrics['pop_2034']:.1f}K người
        - Thay đổi tuyệt đối: {metrics['change_absolute']:+.1f}K người
        - Dân số bình quân 2025-2034: {metrics['avg_forecast_pop']:.1f}K người
        """)

# ============================================================================
# XAI - LIME EXPLANATION
# ============================================================================

if XAI_AVAILABLE and model is not None and lime_training_data is not None:
    st.markdown("---")
    st.subheader("🔍 XAI - LIME (Local + Global)")
    st.markdown("**Giải thích: Những nhân tố nào ảnh hưởng đến dự báo dân số?**")

    # Get window 2014-2024 (11 years) for LIME
    window_2014_2024 = get_window_2014_2024_scaled(selected_area)

    if window_2014_2024 is not None:
        with st.spinner("⏳ Đang tạo giải thích LIME..."):
            lime_result = explain_vietnam_area(selected_area)

        if lime_result is not None:
            # Create two columns for Local and Global
            col_lime_local, col_lime_global = st.columns(2)

            # === LOCAL WEIGHTS ===
            with col_lime_local:
                st.markdown("**📊 Local Weights: Đóng góp của từng năm (2014-2024)**")
                
                df_local = lime_result.local_weights.copy()
                
                if not df_local.empty:
                    # Highlight recent years (2023, 2024)
                    df_local["highlight"] = df_local["feature"].isin(["lag_11_2024", "lag_10_2023"])
                    colors = np.where(df_local["highlight"], "#ff6b6b", "#4c72b0")

                    fig_local = go.Figure(
                        data=[
                            go.Bar(
                                x=df_local["weight"],
                                y=df_local["feature"],
                                orientation="h",
                                marker_color=colors,
                                text=df_local["weight"].round(3),
                                textposition="auto",
                                hovertemplate="<b>%{y}</b><br>Trọng số: %{x:.4f}<extra></extra>",
                            )
                        ]
                    )

                    fig_local.update_layout(
                        height=400,
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=40, b=20),
                        title=f"LIME Cục bộ: Tác động 11 năm (2014-2024) → Dự báo 2025",
                        xaxis_title="Trọng số (Local Impact)",
                        yaxis_title="Năm",
                        hovermode="closest",
                    )
                    fig_local.update_yaxes(autorange="reversed")
                    
                    st.plotly_chart(fig_local, use_container_width=True)

                    st.info("🔴 **Năm được highlight**: 2023-2024 (ảnh hưởng trực tiếp đến dự báo 2025)")
                else:
                    st.warning("Không thể tính toán Local weights")

            # === GLOBAL WEIGHTS ===
            with col_lime_global:
                st.markdown("**🌍 Global Weights: Tác động trung bình trên toàn bộ 63 tỉnh**")
                
                df_global = lime_result.global_abs_weights.copy()
                
                if not df_global.empty:
                    # Show top features in global weights
                    df_global_top = df_global.head(11).copy()
                    
                    st.dataframe(
                        df_global_top,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "feature": st.column_config.TextColumn("Năm"),
                            "mean_abs_weight": st.column_config.NumberColumn(
                                "Trọng số Trung bình", 
                                format="%.4f"
                            ),
                        }
                    )
                    
                    st.caption(
                        "Giá trị cao = Năm này có ảnh hưởng quan trọng đến dự báo trên toàn bộ các vùng. "
                        "Năm gần đây (2023-2024) thường có trọng số cao nhất."
                    )
                else:
                    st.warning("Không thể tính toán Global weights")

        else:
            st.warning("⚠️ Không thể tạo giải thích LIME. Vui lòng kiểm tra dữ liệu hoặc mô hình.")
    else:
        st.warning("⚠️ Dữ liệu chuỗi không đủ để giải thích LIME (cần dữ liệu 2014-2024).")

    # === AUTO INSIGHTS ===
    st.markdown("---")
    st.markdown("**💡 Nhận xét tự động**")
    
    years_hist, values_hist = get_historical_data(selected_area)
    if values_hist and len(values_hist) >= 2:
        recent_trend = "tăng" if values_hist[-1] > values_hist[-2] else "giảm"
        st.info(
            f"**{selected_area}** hiện đang có xu hướng **{recent_trend}** trong giai đoạn gần đây. "
            f"Mô hình dự báo sẽ dựa vào tất cả 11 năm dữ liệu lịch sử (2014-2024) để đưa ra dự báo tương lai. "
            f"Các năm gần đây (đặc biệt 2023-2024) thường có ảnh hưởng lớn hơn các năm xa xôi."
        )

else:
    st.markdown("---")
    st.warning("⚠️ **Tính năng XAI-LIME không khả dụng**: Mô hình hoặc dữ liệu LIME chưa được tải.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px;">
    Ứng dụng Demo: Dự báo Dân số Việt Nam 10 năm (2025-2034)
    <br>Dữ liệu: 63 Tỉnh/Thành phố | Mô hình: Dual-CBA (CNN-BiLSTM + Attention) + XAI-LIME
    <br><small>Cập nhật: 2026</small>
</div>
""", unsafe_allow_html=True)
