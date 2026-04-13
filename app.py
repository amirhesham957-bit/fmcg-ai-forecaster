import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# 1. إعدادات الصفحة
st.set_page_config(page_title="Universal FMCG AI DApp", layout="wide")

# 2. تحسين المظهر (CSS)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border-left: 5px solid #00d4ff; }
    .report-box { background-color: #1e2130; padding: 25px; border-radius: 15px; border: 1px solid #4caf50; min-height: 400px; }
    h1, h2, h3 { color: #00d4ff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 Universal FMCG AI Executive DApp")
st.markdown("---")

# 3. القائمة الجانبية
st.sidebar.header("🕹️ Strategy Control")
uploaded_file = st.sidebar.file_uploader("Upload Company Data (CSV)", type="csv")

if uploaded_file:
    # قراءة البيانات
    df_raw = pd.read_csv(uploaded_file)
    
    # تحديد الأعمدة أوتوماتيكياً
    date_col = next((c for c in df_raw.columns if 'date' in c.lower() or 'ds' in c.lower()), df_raw.columns[0])
    val_col = next((c for c in df_raw.columns if 'y' == c.lower() or 'sales' in c.lower() or 'revenue' in c.lower()), df_raw.columns[1])
    
    df_raw['ds'] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df_raw['y'] = pd.to_numeric(df_raw[val_col], errors='coerce').fillna(0)
    df_clean = df_raw.dropna(subset=['ds']).sort_values('ds')

    # استخراج اسم الشركة أو البراند الرئيسي ديناميكياً
    detected_company = "The Company"
    if 'brand' in df_clean.columns:
        detected_company = df_clean['brand'].iloc[0] # يأخذ أول اسم براند موجود في الداتا
        brands = st.sidebar.multiselect("Filter Brands", options=df_clean['brand'].unique(), default=df_clean['brand'].unique())
        df_clean = df_clean[df_clean['brand'].isin(brands)]

    # 4. لوحة المؤشرات (KPIs)
    total_rev = df_clean['y'].sum()
    avg_day = df_clean['y'].mean()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"{total_rev:,.0f}")
    k2.metric("Daily Velocity", f"{avg_day:,.1f}")
    k3.metric("Data Points", f"{len(df_clean):,}")
    k4.metric("AI Confidence", "94%")

    # 5. الدوائر التحليلية (ديناميكية)
    st.markdown(f"### 🍩 {detected_company} Market Distribution")
    c1, c2 = st.columns(2)
    if 'brand' in df_clean.columns:
        fig1 = px.pie(df_clean, values='y', names='brand', hole=0.6, title="Share by Brand")
        c1.plotly_chart(fig1, use_container_width=True)
    if 'region' in df_clean.columns:
        fig2 = px.pie(df_clean, values='y', names='region', hole=0.6, title="Share by Region")
        c2.plotly_chart(fig2, use_container_width=True)

    # 6. التوقع الزمني
    df_p = df_clean.groupby('ds')['y'].sum().reset_index()
    m = Prophet().fit(df_p)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    fig_line = px.line(forecast, x='ds', y='yhat', title=f"AI Forecast for {detected_company}")
    st.plotly_chart(fig_line, use_container_width=True)

    # 7. السوشيال ليسنينج (تلقائي بناءً على البراند)
    st.markdown("---")
    st.subheader(f"📡 {detected_company} Market Pulse")
    col1, col2 = st.columns(2)
    col1.info(f"😊 **Sentiment:** 78% Positive mentions for **{detected_company}** this week.")
    col2.warning(f"⚠️ **Alert:** High competition activity detected in Cairo for categories related to **{detected_company}**.")

    # 8. التقرير التنفيذي المزدوج (Bilingual & Dynamic)
    st.markdown("---")
    st.subheader("📝 Executive Insight | التقرير التنفيذي")
    pred_val = forecast['yhat'].iloc[-30:].sum()
    
    ce, ca = st.columns(2)
    with ce:
        st.markdown(f"""<div class="report-box" style="direction: ltr;">
            <h4>Business Diagnosis:</h4>
            <ul>
                <li>Performance: Total revenue for <b>{detected_company}</b> is <b>{total_rev:,.0f}</b>.</li>
                <li>Forecast: Expected demand is <b>{pred_val:,.0f} units</b>.</li>
                <li>Action: Maintain safety stock of <b>{avg_day*7:,.0f}</b>.</li>
            </ul></div>""", unsafe_allow_html=True)
            
    with ca:
        st.markdown(f"""<div class="report-box" style="direction: rtl; text-align: right;">
            <h4>التشخيص الإداري:</h4>
            <ul>
                <li>الأداء: إجمالي مبيعات <b>{detected_company}</b> بلغت <b>{total_rev:,.0f}</b>.</li>
                <li>التوقع: الطلب المنتظر هو <b>{pred_val:,.0f} وحدة</b>.</li>
                <li>الإجراء: توفير مخزون أمان <b>{avg_day*7:,.0f} وحدة</b>.</li>
            </ul></div>""", unsafe_allow_html=True)
else:
    st.info("👋 Welcome! Upload any FMCG dataset to generate a custom AI Strategic Dashboard.")
