from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow import keras

try:
    import folium
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover
    folium = None
    st_folium = None

from data_processing import (
    YEAR_END,
    YEAR_START,
    fit_global_minmax,
    inverse_transform_series,
    load_erp_csv,
)
from model import BahdanauAttention, build_dual_cba, rolling_forecast
from xai_lime import build_lime_training_data_for_2007, explain_area_window


st.set_page_config(
    page_title="Bản đồ Tương tác Quy hoạch Hạ tầng & XAI",
    layout="wide",
)


ART_DIR = Path("artifacts")

WINDOW_SIZE = 11
FORECAST_YEARS = list(range(2017, 2027))

STATE_CENTROIDS = {
    "New South Wales": (-31.2532183, 146.921099),
    "Victoria": (-37.020100, 144.964600),
    "Queensland": (-20.917574, 142.702789),
    "South Australia": (-30.000233, 136.209152),
    "Western Australia": (-25.042261, 117.793221),
    "Tasmania": (-41.640079, 146.315918),
    "Northern Territory": (-19.491411, 132.550964),
    "Australian Capital Territory": (-35.473469, 149.012375),
    "Other Territories": (-10.0, 133.0),
}


def auto_insights(state: str) -> str:
    if state in ["New South Wales", "Queensland"]:
        return (
            "Động lực tăng trưởng: Lực đẩy chủ yếu đến từ các lag gần nhất (2005-2006), "
            "mô hình dựa mạnh vào quán tính tăng trưởng gần hiện tại"
        )
    if state in ["Victoria", "Western Australia"]:
        return (
            "Có hiệu ứng điều chỉnh âm từ các biến trễ gần do chuỗi dân số đang nằm trong vùng "
            "giá trị trung bình-thấp"
        )
    if state in ["South Australia", "Tasmania"]:
        return "Cấu trúc động học khác biệt: Nhóm lag trung hạn (2000-2004) đóng vai trò điều chỉnh rõ rệt"
    return "Mô hình dự báo thận trọng với các chuỗi biến động nhỏ do lag gần nhất có trọng số âm mạnh"


def growth_alert(actual_2016: float, forecast_2026: float) -> Optional[str]:
    if actual_2016 <= 0:
        return None
    growth_10y = (forecast_2026 / actual_2016) - 1.0
    if growth_10y >= 0.15:
        return (
            "Dân số tăng nhanh: Cần ưu tiên ngân sách mở rộng cơ sở hạ tầng công cộng như bệnh viện, "
            "trường học và hệ thống giao thông tại khu vực này"
        )
    return None


def plot_forecast(actual_years: np.ndarray, actual_vals: np.ndarray, fc_years: List[int], fc_vals: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_years, y=actual_vals, mode="lines+markers", name="Thực tế (1991–2016)"))
    fig.add_trace(
        go.Scatter(
            x=fc_years,
            y=fc_vals,
            mode="lines+markers",
            name="Dự báo cuộn 10 năm (2017–2026)",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Năm",
        yaxis_title="Dân số (Inverse Global Min-Max)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(dtick=5)
    return fig


def render_map(selected_state: str) -> str:
    if folium is None or st_folium is None:
        st.info("Không có `folium/streamlit-folium` — dùng dropdown để chọn khu vực.")
        return selected_state

    m = folium.Map(location=[-25.5, 134.0], zoom_start=4, tiles="cartodbpositron")
    for state, (lat, lon) in STATE_CENTROIDS.items():
        is_sel = state == selected_state
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 if is_sel else 7,
            color="#d62728" if is_sel else "#1f77b4",
            fill=True,
            fill_opacity=0.9,
            tooltip=state,
            popup=state,
        ).add_to(m)
    out = st_folium(m, height=420, use_container_width=True)
    clicked = None
    if isinstance(out, dict):
        clicked = out.get("last_object_clicked_popup") or out.get("last_object_clicked_tooltip")
    return str(clicked) if clicked in STATE_CENTROIDS else selected_state


@st.cache_resource(show_spinner=False)
def load_or_init_artifacts():
    df = load_erp_csv("Australia_SmallAreas.csv")
    # Prefer offline-trained artifacts (train_eval.py). If not found, fall back to quick init.
    scaler_path = ART_DIR / "global_minmax_scaler.joblib"
    model_path = ART_DIR / "dual_cba_best.keras"

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = fit_global_minmax(df)

    if model_path.exists():
        model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"BahdanauAttention": BahdanauAttention},
        )
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    else:
        model = build_dual_cba(window_size=WINDOW_SIZE)

    # LIME training matrix for global distribution
    lime_train = build_lime_training_data_for_2007(df, scaler)
    return df, scaler, model, lime_train


def main():
    st.title("Bản đồ Tương tác Quy hoạch Hạ tầng & XAI")
    st.caption("Dual‑CBA (Dual‑stream CNN‑BiLSTM + Additive Attention) + LIME | ERP Australia 1991–2016")

    df, scaler, model, lime_train = load_or_init_artifacts()

    left, right = st.columns([0.43, 0.57], gap="large")

    with left:
        st.subheader("Bản đồ & Bộ lọc SA2")
        states = sorted(df["STATE_TERRITORY_NAME"].unique().tolist())
        if "Other Territories" not in states:
            states.append("Other Territories")
        state = st.selectbox("Chọn bang/territory", options=states, index=0)
        state = render_map(state)

        df_state = df[df["STATE_TERRITORY_NAME"] == state].copy()
        sa2_tbl = df_state[["SA2_NAME_2011", "SA2_MAINCODE_2011"]].drop_duplicates().sort_values("SA2_NAME_2011")
        sa2_label_to_code = {
            f"{r.SA2_NAME_2011} ({int(r.SA2_MAINCODE_2011)})": int(r.SA2_MAINCODE_2011)
            for r in sa2_tbl.itertuples(index=False)
        }
        sa2_label = st.selectbox("Chọn khu vực SA2", options=list(sa2_label_to_code.keys()))
        sa2_code = sa2_label_to_code[sa2_label]
        sa2_name = sa2_label.split(" (")[0]

        st.markdown("---")
        st.write({"STATE/TERRITORY": state, "SA2_NAME_2011": sa2_name, "SA2_MAINCODE_2011": sa2_code})

        if not (ART_DIR / "dual_cba_best.keras").exists():
            st.warning(
                "Chưa tìm thấy model đã huấn luyện trong `artifacts/`. "
                "Để đúng quy trình end-to-end, hãy chạy `python train_eval.py` trước (BayesianOpt 30 trials)."
            )

    with right:
        st.subheader("Dự báo cuộn 10 năm & Cảnh báo Hạ tầng")

        years = np.arange(YEAR_START, YEAR_END + 1, dtype=int)
        row = df.loc[df["SA2_MAINCODE_2011"] == sa2_code].iloc[0]
        y_raw = np.array([float(row[str(y)]) for y in years], dtype=np.float32)
        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

        # Forecast 2017..2026 using last window ending 2016
        seed = y_scaled[-WINDOW_SIZE:]
        fc_scaled = rolling_forecast(model, seed, horizon=10).numpy()

        y_inv = inverse_transform_series(scaler, y_scaled)
        fc_inv = inverse_transform_series(scaler, fc_scaled)

        fig = plot_forecast(years, y_inv, FORECAST_YEARS, fc_inv)
        fig.update_layout(title=f"{sa2_name} | {state}")
        st.plotly_chart(fig, use_container_width=True)

        msg = growth_alert(actual_2016=float(y_inv[-1]), forecast_2026=float(fc_inv[-1]))
        if msg:
            st.warning(msg)
        else:
            st.success("Tăng trưởng dự báo ổn định (không kích hoạt cảnh báo tăng vọt).")

        st.markdown("---")
        st.subheader("XAI - LIME (Local + Global)")

        # Explain using fixed lags 1996..2006 -> predict 2007 (so we can highlight lag_11_2006, lag_10_2005)
        window_1996_2006 = y_scaled[5:16]
        with st.spinner("Đang tạo giải thích LIME..."):
            lime_res = explain_area_window(
                model=model,
                area_window_1996_2006_scaled=window_1996_2006,
                lime_training_matrix=lime_train,
                num_features=11,
                global_samples=300,
            )

        # Local bar chart
        df_local = lime_res.local_weights.copy()
        if not df_local.empty:
            df_local["highlight"] = df_local["feature"].isin(["lag_11_2006", "lag_10_2005"])
            colors = np.where(df_local["highlight"], "#d62728", "#1f77b4")
            fig2 = go.Figure(
                data=[go.Bar(x=df_local["weight"], y=df_local["feature"], orientation="h", marker_color=colors)]
            )
            fig2.update_layout(
                height=360,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
                title="LIME Local: đóng góp 11 lags (window 1996–2006 → dự báo 2007)",
                xaxis_title="Trọng số (local)",
                yaxis_title="Lag feature",
            )
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)

        # Global weights table (mean abs weight)
        st.markdown("**Trọng số toàn cục (mean |weight| trên nhiều vùng)**")
        st.dataframe(lime_res.global_abs_weights.head(11), use_container_width=True, hide_index=True)

        st.markdown("**Auto-Insights**")
        st.write(auto_insights(state))


if __name__ == "__main__":
    main()

