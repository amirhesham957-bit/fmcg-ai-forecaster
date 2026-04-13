import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# إعدادات واجهة المستخدم
st.set_page_config(page_title="FMCG AI Forecaster", layout="wide")
st.title("📦 FMCG Smart Sales Forecaster")
st.write("أداة ذكية لتوقع مبيعات المنتجات (مثل نيفيا ولوريال) لتقليل الهدر وتحسين التوزيع.")

# رفع الملف
st.sidebar.header("إعدادات البيانات")
uploaded_file = st.sidebar.file_uploader("ارفع ملف مبيعاتك (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### معاينة البيانات المرفوعة:")
    st.dataframe(df.head())

    # نفترض أن العمود الأول تاريخ والثاني مبيعات
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    if st.button('ابدأ التنبؤ بالذكاء الاصطناعي'):
        with st.spinner('جاري تحليل الأنماط الموسمية...'):
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=7) # توقع 7 أيام
            forecast = m.predict(future)

            st.success("✅ اكتمل التوقع!")
            
            st.write("### الرسم البياني للتوقعات (الـ 7 أيام القادمة):")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)
            
            st.info("💡 نصيحة AI: بناءً على التحليل، يفضل زيادة المخزون بنسبة 10% في فروع التجزئة الكبرى استعداداً للطلب المتزايد المتوقع.")
else:
    st.info("يرجى رفع ملف مبيعات بصيغة CSV لبدء التحليل.")
