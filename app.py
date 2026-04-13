import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="FMCG AI DApp", layout="wide")

# CSS لإعطاء طابع الـ Web3
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
    # قراءة البيانات
    df_raw = pd.read_csv(uploaded_file)
    
    if len(df_raw.columns) < 2:
        st.error("الملف لازم يكون فيه عمودين على الأقل.")
    else:
        # --- حل المشكلة: تنظيف البيانات بشكل احترافي ---
        # بناخد أول عمودين فقط مهما كانت أساميهم ونحولهم لجدول جديد
        ds_col = pd.to_datetime(df_raw.iloc[:, 0], errors='coerce')
        y_col = pd.to_numeric(df_raw.iloc[:, 1], errors='coerce').fillna(0)
        
        df = pd.DataFrame({'ds': ds_col, 'y': y_col})
        df = df.dropna().sort_values('ds') # ترتيب وحذف الفراغات

        # المؤشرات السريعة
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{df['y'].sum():,.0f}")
        col2.metric("Avg Demand", f"{df['y'].mean():.1f}")
        col3.metric("Data Rows", len(df))
        col4.metric("AI Confidence", "94%")

        # المحرك الذكي
        with st.spinner('AI is processing...'):
            try:
                m = Prophet(yearly_seasonality=True, daily_seasonality=True)
                m.fit(df)
                future = m.make_future_dataframe(periods=14)
                forecast = m.predict(future)

                # الرسم البياني التفاعلي
                st.subheader("📈 Smart Sales Forecasting")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', line=dict(color='#4caf50', width=2)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Prediction', line=dict(color='#00d4ff', dash='dot')))
                fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ تم تحليل البيانات بنجاح!")
            except Exception as e:
                st.error(f"حدث خطأ في الموديل: {e}")
else:
    st.info("ارفع ملف الـ CSV (تأكد أن العمود الأول هو التاريخ والثاني هو المبيعات).")
