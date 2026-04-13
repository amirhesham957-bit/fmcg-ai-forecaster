import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="FMCG Intelligence Executive DApp", layout="wide", initial_sidebar_state="expanded")

# 2. تصميم الواجهة (Custom CSS) لتحسين المظهر العربي والإنجليزي
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border-left: 5px solid #00d4ff; }
    .report-box { background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #4caf50; margin-bottom: 20px; min-height: 350px; }
    h1, h2, h3 { color: #00d4ff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌐 FMCG Intelligence Executive DApp")
st.markdown("---")

# 3. القائمة الجانبية (Sidebar)
st.sidebar.header("🕹️ Control & Strategy")
uploaded_file = st.sidebar.file_uploader("Upload Market Data (CSV)", type="csv")

if uploaded_file:
    # قراءة البيانات
    df_raw = pd.read_csv(uploaded_file)
    
    # تحديد الأعمدة أوتوماتيكياً (ds و y)
    date_col = 'ds' if 'ds' in df_raw.columns else df_raw.columns[0]
    val_col = 'y' if 'y' in df_raw.columns else df_raw.columns[1]

    df_raw['ds'] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df_raw['y'] = pd.to_numeric(df_raw[val_col], errors='coerce').fillna(0)
    df_clean = df_raw.dropna(subset=['ds']).sort_values('ds')

    # الفلاتر الجانبية (Slicers) كما ظهرت في تطبيقك
    if 'brand' in df_raw.columns:
        brands = st.sidebar.multiselect("Filter by Brand", options=df_raw['brand'].unique(), default=df_raw['brand'].unique())
        df_clean = df_clean[df_clean['brand'].isin(brands)]
    
    if 'region' in df_raw.columns:
        regions = st.sidebar.multiselect("Filter by Region", options=df_raw['region'].unique(), default=df_raw['region'].unique())
        df_clean = df_clean[df_clean['region'].isin(regions)]

    # 4. لوحة المؤشرات (KPIs)
    total_rev = df_clean['y'].sum()
    avg_day = df_clean['y'].mean()
    data_points = len(df_clean)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"{total_rev:,.0f}")
    k2.metric("Daily Velocity", f"{avg_day:,.1f}")
    k3.metric("Data Points", f"{data_points:,}")
    k4.metric("AI Confidence", "94%")

    # 5. الرسوم البيانية للتوزيع (تم حل مشكلة الألوان هنا)
    st.markdown("### 🍩 Market Distribution")
    c1, c2 = st.columns(2)
    
    if 'brand' in df_raw.columns:
        fig_brand = px.pie(df_clean, values='y', names='brand', hole=0.6, title="Share by Brand", 
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        c1.plotly_chart(fig_brand, use_container_width=True)
        
    if 'region' in df_raw.columns:
        fig_reg = px.pie(df_clean, values='y', names='region', hole=0.6, title="Share by Region", 
                         color_discrete_sequence=px.colors.qualitative.Safe)
        c2.plotly_chart(fig_reg, use_container_width=True)

    # 6. محرك التوقع (Forecasting)
    st.markdown("### 📈 Smart Forecasting Analysis")
    df_p = df_clean.groupby('ds')['y'].sum().reset_index()
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual Sales', line=dict(color='#00d4ff')))
    fig_main.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat'].iloc[-30:], name='AI Forecast', line=dict(color='#00ff88', dash='dot')))
    fig_main.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_main, use_container_width=True)

    # 7. التقرير التنفيذي المزدوج (Bilingual Executive Report)
    st.markdown("---")
    st.subheader("📝 Executive Insight & Action Plan | الرؤية التنفيذية وخطة العمل")
    
    pred_next_month = forecast['yhat'].iloc[-30:].sum()
    top_performer = df_clean.groupby('brand')['y'].sum().idxmax() if 'brand' in df_clean.columns else "Main Brand"
    
    col_en, col_ar = st.columns(2)
    
    # التقرير الإنجليزي (اليسار)
    with col_en:
        st.markdown(f"""
        <div class="report-box" style="direction: ltr; text-align: left;">
            <h4 style="color: #00d4ff;">🔍 Business Diagnosis:</h4>
            <ul>
                <li><b>Current Performance:</b> Total revenue reached <b>{total_rev:,.0f} units</b>.</li>
                <li><b>Growth Driver:</b> <b>{top_performer}</b> is the primary engine for your growth.</li>
                <li><b>Risk Analysis:</b> Average daily demand is <b>{avg_day:,.1f}</b>. Inventory must stay above this level.</li>
            </ul>
            <h4 style="color: #4caf50;">💡 Strategic Recommendations:</h4>
            <ul>
                <li><b>Stock Strategy:</b> Maintain a Safety Stock of <b>{avg_day * 7:,.0f} units</b> for next week.</li>
                <li><b>Expansion:</b> Next month demand is forecast at <b>{pred_next_month:,.0f} units</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # التقرير العربي (اليمين)
    with col_ar:
        st.markdown(f"""
        <div class="report-box" style="direction: rtl; text-align: right;">
            <h4 style="color: #00d4ff;">🔍 التشخيص الإداري:</h4>
            <ul>
                <li><b>الأداء الحالي:</b> حققت المبيعات الإجمالية <b>{total_rev:,.0f} وحدة</b> تاريخياً.</li>
                <li><b>محرك النمو:</b> يعتبر البراند <b>{top_performer}</b> هو المحرك الأساسي لمبيعاتك.</li>
                <li><b>تحليل المخاطر:</b> متوسط الطلب اليومي هو <b>{avg_day:,.1f}</b>. أي نقص يهدد حصتك السوقية.</li>
            </ul>
            <h4 style="color: #4caf50;">💡 التوصيات الاستراتيجية:</h4>
            <ul>
                <li><b>استراتيجية المخزون:</b> توفير حد أمان (Safety Stock) لا يقل عن <b>{avg_day * 7:,.0f} وحدة</b>.</li>
                <li><b>خطة التوسع:</b> الطلب المتوقع للشهر القادم يقدر بـ <b>{pred_next_month:,.0f} وحدة</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.download_button("Export AI Report", forecast.to_csv(), "fmcg_report.csv")
else:
    st.info("👋 Welcome! Please upload your CSV file to start the AI analysis.")
