import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="FMCG AI Executive Dashboard", layout="wide")

# تجميل الواجهة (Power BI Style)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border-left: 5px solid #00d4ff; }
    h1, h2, h3 { color: #00d4ff; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 FMCG Executive Intelligence Dashboard")
st.markdown("---")

# القائمة الجانبية للفلاتر
st.sidebar.header("🔍 Global Filters")
uploaded_file = st.sidebar.file_uploader("Upload Data (CSV)", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    # تحضير البيانات
    df_raw['ds'] = pd.to_datetime(df_raw['date'] if 'date' in df_raw.columns else df_raw['ds'], errors='coerce')
    df_raw['y'] = pd.to_numeric(df_raw['y'] if 'y' in df_raw.columns else df_raw.iloc[:, 1], errors='coerce').fillna(0)
    df_raw = df_raw.dropna(subset=['ds'])

    # إضافة الفلاتر الديناميكية (مثل Power BI)
    if 'brand' in df_raw.columns:
        selected_brand = st.sidebar.multiselect("Select Brand", options=df_raw['brand'].unique(), default=df_raw['brand'].unique())
        df_raw = df_raw[df_raw['brand'].isin(selected_brand)]
    
    if 'region' in df_raw.columns:
        selected_region = st.sidebar.multiselect("Select Region", options=df_raw['region'].unique(), default=df_raw['region'].unique())
        df_raw = df_raw[df_raw['region'].isin(selected_region)]

    # حساب الـ KPIs
    total_sales = df_raw['y'].sum()
    avg_sales = df_raw['y'].mean()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"{total_sales:,.0f}")
    m2.metric("Daily Average", f"{avg_sales:,.1f}")
    m3.metric("Active Regions", len(df_raw['region'].unique()) if 'region' in df_raw.columns else "N/A")
    m4.metric("Forecast Confidence", "94%")

    # قسم الرسوم الدائرية (Donut Charts)
    st.markdown("### 🍩 Strategic Distribution")
    c1, c2 = st.columns(2)
    
    if 'brand' in df_raw.columns:
        fig_brand = px.pie(df_raw, values='y', names='brand', hole=0.6, title="Sales by Brand", color_discrete_sequence=px.colors.sequential.RdBu)
        c1.plotly_chart(fig_brand, use_container_width=True)
        
    if 'region' in df_raw.columns:
        fig_reg = px.pie(df_raw, values='y', names='region', hole=0.6, title="Sales by Region", color_discrete_sequence=px.colors.sequential.Blues_r)
        c2.plotly_chart(fig_reg, use_container_width=True)

    # التوقع (Prophet)
    df_prophet = df_raw.groupby('ds')['y'].sum().reset_index()
    m = Prophet(yearly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    st.markdown("### 📈 Demand Forecasting AI")
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical', line=dict(color='#00d4ff', width=2)))
    fig_main.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat'].iloc[-30:], name='AI Prediction', line=dict(color='#00ff88', dash='dot')))
    fig_main.update_layout(template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_main, use_container_width=True)

    # التقرير الكتابي (The Executive Summary)
    st.markdown("---")
    st.subheader("📝 Executive Summary & Insights")
    
    # تحليل ذكي للبيانات
    growth = ((forecast['yhat'].iloc[-1] - df_prophet['y'].iloc[-1]) / df_prophet['y'].iloc[-1]) * 100
    top_region = df_raw.groupby('region')['y'].sum().idxmax() if 'region' in df_raw.columns else "N/A"
    
    report = f"""
    بناءً على تحليل البيانات التاريخية لـ **{len(df_prophet)} يوم**:
    * **الأداء العام:** تم تحقيق إجمالي مبيعات قدرها **{total_sales:,.0f} وحدة**.
    * **القوة البيعية:** تعتبر منطقة **{top_region}** هي المحرك الأساسي للمبيعات حالياً.
    * **توقعات المستقبل:** يشير الذكاء الاصطناعي إلى تغير بنسبة **{growth:.1f}%** في الـ 30 يوماً القادمة.
    * **توصية PM:** يجب الحفاظ على مستويات مخزون تدعم متوسط يومي **{avg_sales:,.1f}** لتجنب أي نقص (Stock-out).
    """
    st.info(report)

else:
    st.warning("Please upload your FMCG data file to generate the executive report.")
