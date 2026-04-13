import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="FMCG AI DApp", layout="wide")

# ستايل احترافي
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
    
    # خطوة تنظيف البيانات (حل المشكلة)
    df.columns = ['ds', 'y'] + list(df.columns[2:])
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    # تحويل أي قيمة غير رقمية في y إلى أرقام، وما يفشل تحويله يصبح NaN ثم صفر
    df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
    # حذف الصفوف اللي فيها تواريخ غلط
    df = df.dropna(subset=['ds'])

    # المؤشرات السريعة (الآن ستعمل بدون خطأ)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"{df['y'].sum():,.0f}")
    col2.metric("Avg Demand", f"{df['y'].mean():.1;f}")
    col3.metric("Data Rows", len(df))
    col4.metric("System Status", "Live")

    # المحرك الذكي
    with st.spinner('AI is analyzing data...'):
        m = Prophet(yearly_seasonality=True, daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=14)
        forecast = m.predict(future)

    # الرسم البياني
    st.subheader("📈 AI Forecasting Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', line=dict(color='#4caf50')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Prediction', line=dict(color='#00d4ff', dash='dot')))
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ Analysis Complete!")
else:
    st.info("Please upload the modified CSV file (with ds and y columns).")
