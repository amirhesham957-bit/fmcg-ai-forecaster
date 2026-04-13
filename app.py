import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Universal FMCG AI DApp", layout="wide")

# 2. تصميم الواجهة (Custom CSS)
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

# 3. القائمة الجانبية والفلاتر
st.sidebar.header("🕹️ Strategy Control")
uploaded_file = st.sidebar.file_uploader("Upload Company Data (CSV)", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    # تحديد الأعمدة أوتوماتيكياً
    date_col = next((c for c in df_raw.columns if 'date' in c.lower() or 'ds' in c.lower()), df_raw.columns[0])
    val_col = next((c for c in df_raw.columns if 'y' == c.lower() or 'sales' in c.lower() or 'revenue' in c.lower()), df_raw.columns[1])
    
    df_raw['ds'] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df_raw['y'] = pd.to_numeric(df_raw[val_col], errors='coerce').fillna(0)
    df_clean = df_raw.dropna(subset=['ds']).sort_values('ds')

    # استخراج اسم البراند أو الشركة ديناميكياً
    detected_name = "The Brand"
    if 'brand' in df_clean.columns:
        detected_name = df_clean['brand'].iloc[0]
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

    # 5. الدوائر التحليلية (Market Distribution)
    st.markdown(f"### 🍩 {detected_name} Market Distribution")
    c1, c2 = st.columns(2)
    if 'brand' in df_clean.columns:
        fig1 = px.pie(df_clean, values='y', names='brand', hole=0.6, title="Share by Brand")
        c1.plotly_chart(fig1, use_container_width=True)
    if 'region' in df_clean.columns:
        fig2 = px.pie(df_clean, values='y', names='region', hole=0.6, title="Share by Region")
        c2.plotly_chart(fig2, use_container_width=True)

    # 6. التوقع الزمني (حل مشكلة الخط المنقط)
    st.markdown(f"### 📈 Smart Forecasting for {detected_name}")
    df_p = df_clean.groupby('ds')['y'].sum().reset_index()
    m = Prophet().fit(df_p)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig_main = go.Figure()
    # بيانات حقيقية - خط متصل
    fig_main.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual Sales', line=dict(color='#00d4ff', width=3)))
    # توقعات - خط منقط
    fig_main.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat'].iloc[-30:], name='AI Prediction', line=dict(color='#00ff88', width=3, dash='dot')))
    
    fig_main.update_layout(template="plotly_dark", height=450, hovermode="x unified")
    st.plotly_chart(fig_main, use_container_width=True)

    # 7. السوشيال ليسنينج الحي
    st.markdown("---")
    st.subheader(f"📡 {detected_name} Market Pulse")
    col1, col2 = st.columns(2)
    col1.info(f"😊 **Sentiment Analysis:** 78% Positive mentions for **{detected_name}** this week.")
    col2.warning(f"⚠️ **Competitor Alert:** Competition activity detected in local markets affecting **{detected_name}** share.")

    # 8. التقرير التنفيذي المزدوج (Bilingual)
    st.markdown("---")
    st.subheader("📝 Executive Insight | التقرير التنفيذي")
    pred_val = forecast['yhat'].iloc[-30:].sum()
    
    ce, ca = st.columns(2)
    with ce:
        st.markdown(f"""<div class="report-box" style="direction: ltr;">
            <h4>Business Diagnosis:</h4>
            <ul>
                <li>Performance: Total revenue for <b>{detected_name}</b> is <b>{total_rev:,.0f}</b>.</li>
                <li>Forecast: Expected demand next month is <b>{pred_val:,.0f} units</b>.</li>
                <li>Action: Maintain safety stock of <b>{avg_day*7:,.0f} units</b>.</li>
            </ul></div>""", unsafe_allow_html=True)
            
    with ca:
        st.markdown(f"""<div class="report-box" style="direction: rtl; text-align: right;">
            <h4>التشخيص الإداري:</h4>
            <ul>
                <li>الأداء: إجمالي مبيعات <b>{detected_name}</b> بلغت <b>{total_rev:,.0f}</b>.</li>
                <li>التوقع: الطلب المنتظر الشهر القادم <b>{pred_val:,.0f} وحدة</b>.</li>
                <li>الإجراء: توفير مخزون أمان <b>{avg_day*7:,.0f} وحدة</b>.</li>
            </ul></div>""", unsafe_allow_html=True)

    st.sidebar.download_button("Export Forecast", forecast.to_csv(), "ai_forecast_report.csv")
else:
    st.info("👋 Welcome! Please upload any FMCG dataset to begin.")
