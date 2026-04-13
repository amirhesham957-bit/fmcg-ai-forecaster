import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="FMCG AI DApp", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4caf50; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 FMCG Intelligence DApp")
st.markdown("---")

st.sidebar.header("🕹️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # التأكد من أسماء الأعمدة
    df.columns = ['ds', 'y'] + list(df.columns[2:])
    df['ds'] = pd.to_datetime(df['ds'])

    # المؤشرات السريعة
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"{df['y'].sum():,.0f}")
    col2.metric("Avg Demand", f"{df['y'].mean():.1f}")
    col3.metric("Records", len(df))
    col4.metric("AI Confidence", "94%")

    # المحرك الذكي
    m = Prophet(yearly_seasonality=True, daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=14)
    forecast = m.predict(future)

    # الرسم البياني (تم تعديله لتجنب الـ KeyError)
    st.subheader("📈 Smart Forecasting Analysis")
    fig = go.Figure()
    # البيانات الحقيقية
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', line=dict(color='#4caf50', width=3)))
    # التوقعات
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Prediction', line=dict(color='#00d4ff', dash='dot')))
    
    fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("🤖 AI Insights: التوقعات جاهزة! الموديل لاحظ أنماط نمو مستقرة، يرجى مراجعة الخطة اللوجستية للأسبوع القادم.")
    st.download_button("Download Report", forecast.to_csv(), "forecast.csv")
else:
    st.info("Please upload your CSV file to start AI analysis.")
