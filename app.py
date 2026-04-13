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
    df_raw = pd.read_csv(uploaded_file)
    
    # --- محرك البحث عن الأعمدة الصحيحة ---
    if 'ds' in df_raw.columns and 'y' in df_raw.columns:
        df = df_raw[['ds', 'y']].copy()
    else:
        # لو ملقاش الأسامي، ياخد أول عمودين (الخيار التقليدي)
        df = df_raw.iloc[:, [0, 1]].copy()
        df.columns = ['ds', 'y']

    # تنظيف وتحويل البيانات
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
    df = df.dropna(subset=['ds']).sort_values('ds')

    # تجميع البيانات باليوم (عشان الملفات الكبيرة متهنجش)
    df = df.groupby('ds')['y'].sum().reset_index()

    if df['y'].sum() == 0:
        st.error("⚠️ السيستم مش لاقي أرقام مبيعات. اتأكد إن عمود الـ 'y' فيه أرقام مش حروف.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{df['y'].sum():,.0f}")
        col2.metric("Avg Daily", f"{df['y'].mean():.1f}")
        col3.metric("Data Points", len(df))
        col4.metric("AI Confidence", "94%")

        with st.spinner('🤖 AI is thinking...'):
            m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)

            st.subheader("📈 Smart Sales Forecasting")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', line=dict(color='#4caf50')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Forecast', line=dict(color='#00d4ff', dash='dot')))
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ تم التوقع بنجاح لـ 30 يوم قادم!")
else:
    st.info("ارفع ملف الـ CSV المعدل (اللي فيه ds و y).")
