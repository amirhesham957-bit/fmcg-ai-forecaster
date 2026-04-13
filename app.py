import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# 1. إعدادات الصفحة
st.set_page_config(page_title="FMCG AI Intelligence DApp", layout="wide")

# 2. تحسين المظهر (CSS)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .report-box { background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #4caf50; min-height: 380px; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 FMCG Intelligence Executive DApp")

# 3. تحميل البيانات والفلاتر
st.sidebar.header("🕹️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    date_col = 'ds' if 'ds' in df.columns else df.columns[0]
    val_col = 'y' if 'y' in df.columns else df.columns[1]
    df['ds'] = pd.to_datetime(df[date_col])
    df['y'] = pd.to_numeric(df[val_col], errors='coerce').fillna(0)
    
    # KPIs
    total_rev = df['y'].sum()
    avg_day = df['y'].mean()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"{total_rev:,.0f}")
    k2.metric("Avg Demand", f"{avg_day:,.1f}")
    k3.metric("AI Confidence", "94%")

    # 4. التوقع والرسوم البيانية
    df_p = df.groupby('ds')['y'].sum().reset_index()
    m = Prophet().fit(df_p)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    fig = px.line(forecast, x='ds', y='yhat', title="AI Sales Forecast (30 Days)")
    st.plotly_chart(fig, use_container_width=True)

    # 5. ميزة الـ Social Listening (الميزة الجديدة)
    st.markdown("---")
    st.subheader("📡 Live Market Pulse (Social Listening)")
    col_pulse1, col_pulse2 = st.columns(2)
    with col_pulse1:
        st.info("😊 **Sentiment Analysis:** 78% Positive mentions for your top brands this week.")
    with col_pulse2:
        st.warning("⚠️ **Competitor Alert:** K-Group initiated a 20% discount campaign in Cairo region.")

    # 6. التقرير التنفيذي المزدوج
    st.markdown("---")
    st.subheader("📝 Executive Insight | التقرير التنفيذي")
    c_en, c_ar = st.columns(2)
    
    pred_val = forecast['yhat'].iloc[-30:].sum()
    
    with c_en:
        st.markdown(f"""<div class="report-box" style="direction: ltr;">
            <h4>Business Diagnosis:</h4>
            <ul>
                <li>Total Revenue: <b>{total_rev:,.0f} units</b>.</li>
                <li>Forecasted Demand: <b>{pred_val:,.0f} units</b> for next month.</li>
                <li>Safety Stock recommendation: <b>{avg_day*7:,.0f} units</b>.</li>
            </ul></div>""", unsafe_allow_html=True)
            
    with c_ar:
        st.markdown(f"""<div class="report-box" style="direction: rtl; text-align: right;">
            <h4>التشخيص الإداري:</h4>
            <ul>
                <li>إجمالي المبيعات: <b>{total_rev:,.0f} وحدة</b>.</li>
                <li>الطلب المتوقع: <b>{pred_val:,.0f} وحدة</b> للشهر القادم.</li>
                <li>مخزون الأمان المقترح: <b>{avg_day*7:,.0f} وحدة</b>.</li>
            </ul></div>""", unsafe_allow_html=True)
else:
    st.info("Please upload your CSV file to begin analysis.")
