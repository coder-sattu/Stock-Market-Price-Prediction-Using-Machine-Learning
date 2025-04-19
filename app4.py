import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta
import requests

# Page configuration
st.set_page_config(page_title="Stock Market Prediction", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
    }
    .metric-card {
        background: linear-gradient(45deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .prediction-card {
        background: linear-gradient(45deg, #2937f0, #9f1ae2);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .news-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .news-card:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 Stock Market Prediction App")

# Sidebar
st.sidebar.title("📊 Stock Selection")

# Stock selection
stock = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")

# Date range
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Fetch stock data
try:
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the selected stock symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    st.stop()

# Fetch Company Information
stock_info = yf.Ticker(stock).info
st.sidebar.markdown("# 🏢 Company Information")
st.sidebar.markdown(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
st.sidebar.markdown(f"**Industry:** {stock_info.get('industry', 'N/A')}")
st.sidebar.markdown(f"**Market Cap:** ₹{stock_info.get('marketCap', 'N/A'):,}")
st.sidebar.markdown(f"**52 Week High:** ₹{stock_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
st.sidebar.markdown(f"**52 Week Low:** ₹{stock_info.get('fiftyTwoWeekLow', 'N/A'):.2f}")

# Links
yahoo_finance_url = f"https://finance.yahoo.com/quote/{stock}"
st.sidebar.markdown(f"**Links:**")
st.sidebar.markdown(f"• [Yahoo Finance]({yahoo_finance_url})")
company_website = stock_info.get('website', 'N/A')
if company_website != 'N/A':
    st.sidebar.markdown(f"**Website:** [{company_website}]({company_website})")
else:
    st.sidebar.markdown(f"**Website:** {company_website}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "💰 Technical Indicators", "📊 Financial Metrics", "📰 News"])

with tab1:
    # Stock price chart
    st.subheader("Stock Price Analysis")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title(f"{stock} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.legend()
    st.pyplot(fig)

    # Moving averages
    st.subheader("Moving Averages")
    ma_col1, ma_col2, ma_col3 = st.columns(3)
    
    with ma_col1:
        ma50 = data['Close'].rolling(window=50).mean()
        st.metric("50-Day MA", f"₹{float(ma50.iloc[-1]):.2f}")
        
    with ma_col2:
        ma100 = data['Close'].rolling(window=100).mean()
        st.metric("100-Day MA", f"₹{float(ma100.iloc[-1]):.2f}")
        
    with ma_col3:
        ma200 = data['Close'].rolling(window=200).mean()
        st.metric("200-Day MA", f"₹{float(ma200.iloc[-1]):.2f}")

with tab2:
    # Technical Indicators
    st.subheader("Technical Analysis")
    
    # RSI Calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        rsi_value = float(rsi.iloc[-1])
        st.markdown("""
            <div class="metric-card">
                <h4>RSI (14-day)</h4>
                <p style='font-size: 24px;'>{:.2f}</p>
                <p>{}</p>
            </div>
        """.format(rsi_value, 
                  "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"), 
        unsafe_allow_html=True)
        
    with tech_col2:
        st.markdown("""
            <div class="metric-card">
                <h4>Volume Analysis</h4>
                <p>Current: {:,}</p>
                <p>Average: {:,}</p>
            </div>
        """.format(data['Volume'].iloc[-1], data['Volume'].mean()),
        unsafe_allow_html=True)

with tab3:
    # Financial Metrics
    st.subheader("Financial Overview")
    
    fin_col1, fin_col2, fin_col3 = st.columns(3)
    
    with fin_col1:
        st.metric("Market Cap", f"₹{stock_info.get('marketCap')/1e9:.2f}B")
        st.metric("P/E Ratio", stock_info.get('trailingPE', 'N/A'))
        
    with fin_col2:
        st.metric("52W High", f"₹{stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
        st.metric("Volume", f"{stock_info.get('volume', 0):,}")
        
    with fin_col3:
        st.metric("52W Low", f"₹{stock_info.get('fiftyTwoWeekLow', 0):.2f}")
        st.metric("Avg Volume", f"{stock_info.get('averageVolume', 0):,}")

    # Financial Ratios
    st.subheader("📈 Financial Ratios")
    ratios_data = {
        "P/B Ratio": stock_info.get('priceToBook', 'N/A'),
        "Debt to Equity": stock_info.get('debtToEquity', 'N/A'),
        "ROE": stock_info.get('returnOnEquity', 'N/A'),
        "Profit Margin": stock_info.get('profitMargins', 'N/A'),
    }
    st.table(pd.DataFrame([ratios_data]))

with tab4:
    # News Section
    st.subheader("Latest Market News")
    
    # News API configuration
    NEWS_API_KEY = "94b0b867d3e048fe9396ea4661a1c87c"
    news_url = f"https://newsapi.org/v2/everything?q={stock_info.get('longName', '')}&apiKey={NEWS_API_KEY}"
    
    try:
        news_response = requests.get(news_url)
        if news_response.status_code == 200:
            news_data = news_response.json()
            for article in news_data.get('articles', [])[:5]:
                st.markdown(f"""
                    <div class="news-card">
                        <h4>{article['title']}</h4>
                        <p>{article['description']}</p>
                        <a href="{article['url']}" target="_blank">Read more →</a>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Unable to fetch news at the moment.")
    except:
        st.warning("Unable to fetch news at the moment.")

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 30px; padding: 20px;'>
        <p>Made with ❤️ by Your Name</p>
        <p>Data provided by Yahoo Finance</p>
    </div>
""", unsafe_allow_html=True)