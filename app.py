import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime

# 網頁標題與排版設定
st.set_page_config(page_title="AI 股票追蹤與預測儀表板", layout="wide")
st.title("📈 AI 股票追蹤與預測儀表板")

# 側邊欄
st.sidebar.header("設定")
ticker_symbol = st.sidebar.text_input("輸入股票代號 (例如: 2330.TW, AAPL, NVDA)", "2330.TW")

@st.cache_data(ttl=3600)
def load_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    news = stock.news
    # 修正1：拿掉 stock 物件，只回傳可以快取的數據資料
    return hist, news

if ticker_symbol:
    st.write(f"正在分析: **{ticker_symbol}**")
    try:
        # 修正2：這裡也對應拿掉 stock
        hist, news = load_data(ticker_symbol)
        
        if hist.empty:
            st.error("找不到該股票的數據，請確認代號是否正確。")
        else:
            # 今日現況
            st.subheader("📊 今日現況")
            latest_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = latest_close - prev_close
            change_pct = (change / prev_close) * 100
            st.metric(label=f"最新收盤價 ({hist.index[-1].strftime('%Y-%m-%d')})", value=f"{latest_close:.2f}", delta=f"{change:.2f} ({change_pct:.2f}%)")

            # 歷史走勢圖
            st.subheader("📉 歷史走勢圖")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='收盤價'))
            st.plotly_chart(fig_hist, use_container_width=True)

            # 最新新聞
            st.subheader("📰 最新相關新聞")
            for item in news[:5]:
                pub_date = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                st.markdown(f"* **[{pub_date}] [{item['title']}]({item['link']})**")

            # AI 預測
            st.subheader("🤖 AI 模型半年 (180天) 趨勢預測")
            with st.spinner('正在進行 AI 模型訓練與運算，請稍候...'):
                df_prophet = hist.reset_index()[['Date', 'Close']]
                df_prophet['Date'] = df_prophet['Date'].dt.tz_localize(None)
                df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
                model = Prophet(daily_seasonality=False)
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=180)
                forecast = model.predict(future)
                fig_predict = plot_plotly(model, forecast)
                st.plotly_chart(fig_predict, use_container_width=True)
                future_date = forecast['ds'].iloc[-1].strftime('%Y-%m-%d')
                future_price = forecast['yhat'].iloc[-1]
                st.info(f"💡 預計半年後 ({future_date}) 的趨勢預測價格約落在：**{future_price:.2f}**")
    except Exception as e:
        st.error(f"發生錯誤: {e}")
