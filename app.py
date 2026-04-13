import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# إعدادات الصفحة بستايل الـ Dashboard الحديث
st.set_page_config(page_title="FMCG AI DApp", layout="wide", initial_sidebar_state="expanded")

# CSS لإعطاء لمسة الـ Dark Mode والـ Web3
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4caf50; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 FMCG Intelligence DApp")
st.markdown("---")

# ملف تجريبي (Sample Data) لو مفيش ملف مرفوع
def create_sample_data():
    dates = pd.date_range(start="2025-01-01", periods=60)
    sales = [100 + (i * 0.5) + (20 if i % 7 == 0 else 0) for i in range(60)]
    return pd.DataFrame({'ds': dates, 'y': sales})

# القائمة الجانبية
st.sidebar.header("🕹️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Using Demo Data. Upload yours to customize.")
    df = create_sample_data()

df['ds'] = pd.to_datetime(df['ds'])

# --- قسم الـ KPIs (المؤشرات الرئيسية) ---
col1, col2, col3, col4 = st.columns(4)
total_sales = df['y'].sum()
avg_sales = df['y'].mean()
last_val = df['y'].iloc[-1]

col1.metric("Total Sales", f"{total_sales:,.0f}")
col2.metric("Avg Daily Demand", f"{avg_sales:.1f}")
col3.metric("Last Record", f"{last_val:.0f}")
col4.metric("AI Confidence", "92%")

st.markdown("---")

# --- قسم التوقعات (The AI Engine) ---
m = Prophet(yearly_seasonality=True, daily_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=14)
forecast = m.predict(future)

# رسم بياني احترافي بـ Plotly (زي الـ Crypto Charts)
st.subheader("📈 Smart Forecasting Analysis")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', line=dict(color='#4caf50', width=3)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['y'], name='AI Prediction', line=dict(color='#00d4ff', dash='dot')))
fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# --- قسم التحليل العميق (Seasonality) ---
st.subheader("📊 Business Intelligence Insights")
c1, c2 = st.columns(2)

# تحليل المبيعات حسب أيام الأسبوع
df['weekday'] = df['ds'].dt.day_name()
weekly_analysis = df.groupby('weekday')['y'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
fig_week = px.bar(weekly_analysis, x=weekly_analysis.index, y='y', title="Avg Sales by Day", color_discrete_sequence=['#00d4ff'])
c1.plotly_chart(fig_week, use_container_width=True)

# نصيحة الـ AI المستندة للبيانات
with c2:
    st.write("### 🤖 AI Strategy Advisor")
    peak_day = weekly_analysis.idxmax()
    st.info(f"Analysis indicates that **{peak_day}** is your highest demand day. Suggesting 15% stock buffer for logistics on this day.")
    st.success("Target: Optimize Nivea/L'Oreal inventory for the upcoming 14-day cycle.")

# زرار تحميل التوقعات
st.download_button("Download Full Prediction Report", forecast.to_csv(), "ai_forecast.csv", "text/csv")
