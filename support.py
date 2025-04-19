import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import requests
from newsapi import NewsApiClient
from sklearn.preprocessing import MinMaxScaler

# Add currency symbol detection function
def get_currency_symbol(stock_symbol):
    # Check if it's an Indian stock (ends with .NS or .BO)
    if stock_symbol.endswith(('.NS', '.BO')):
        return '₹'
    # For US stocks (default)
    return '$'

# Load Bull & Bear Icons
bull_icon = "https://cdn-icons-png.flaticon.com/512/235/235370.png"
bear_icon = "https://cdn-icons-png.flaticon.com/512/1025/1025379.png"

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stock-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #2937f0, #9f1ae2);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
# Update model path to correct location
model = load_model('C:/Users/satva/OneDrive/new/Stock_Market_Prediction_ML/Stock Predictions Model.keras')

# Add loading animation
with st.spinner('Loading Stock Market Predictor...'):
    st.markdown("""
        <div class="stock-header">
            <h1>📊 Stock Market Predictor</h1>
            <p>Your Intelligent Stock Analysis Platform</p>
        </div>
    """, unsafe_allow_html=True)

# Add progress bar for data loading
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
st.success('Ready to analyze stocks!')

# Stock Input with better styling
col1, col2 = st.columns([2, 1])
with col1:
    stock = st.text_input("🔍 Enter Stock Symbol", "GOOG", 
                         help="Enter the stock symbol (e.g., AAPL for Apple)")
with col2:
    st.image("https://img.icons8.com/color/96/000000/stocks.png", width=100)

start = "2012-01-01"
end = "2025-04-07"

# Fetch Stock Data
data = yf.download(stock, start, end)
st.markdown("### Stock Data")
st.dataframe(data, height=300)

# Fetch Company Information
stock_info = yf.Ticker(stock).info
st.sidebar.markdown("# 🏢 Company Information")
st.sidebar.markdown(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
st.sidebar.markdown(f"**Industry:** {stock_info.get('industry', 'N/A')}")
st.sidebar.markdown(f"**Market Cap:** {stock_info.get('marketCap', 'N/A')}")
st.sidebar.markdown(f"**52 Week High:** {stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
st.sidebar.markdown(f"**52 Week Low:** {stock_info.get('fiftyTwoWeekLow', 'N/A')}")
st.sidebar.markdown(f"**Website:** {stock_info.get('website', 'N/A')}")

# Initialize News API Client (update this section)
newsapi = NewsApiClient(api_key='94b0b867d3e048fe9396ea4661a1c87c')

# Fetch Latest News for Selected Company
company_news = newsapi.get_top_headlines(q=stock, language='en', page_size=5)

# Fetch Global Stock Market News
global_news = newsapi.get_top_headlines(category='business', language='en', page_size=5)

# Sidebar: Company-Specific News
st.sidebar.markdown("# 📰 Latest News for " + stock)
if company_news['articles']:
    for article in company_news['articles']:
        st.sidebar.markdown(f"**{article['title']}**")
        st.sidebar.markdown(f"{article['description']}")
        st.sidebar.markdown(f"[Read more]({article['url']})")
        st.sidebar.markdown("---")
else:
    st.sidebar.markdown("⚠️ No recent news available.")

# Main Page: Global Stock Market News
st.sidebar.markdown("# 🌎 Global Stock Market News")
if global_news['articles']:
    for article in global_news['articles']:
        st.sidebar.markdown(f"**{article['title']}**")
        st.sidebar.markdown(f"{article['description']}")
        st.sidebar.markdown(f"[Read more]({article['url']})")
        st.sidebar.markdown("---")
else:
    st.markdown("⚠️ No recent global market news available.")

# Data Processing
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

# Prepare Data for Model
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Moving Averages
fig, axes = plt.subplots(3, 1, figsize=(6, 4))

ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

axes[0].plot(ma_50_days, 'r', label='MA50')
axes[0].plot(data.Close, 'g', label='Stock Price')
axes[0].set_title("Price vs MA50", fontsize=10)
axes[0].legend()

axes[1].plot(ma_50_days, 'r', label='MA50')
axes[1].plot(ma_100_days, 'b', label='MA100')
axes[1].plot(data.Close, 'g', label='Stock Price')
axes[1].set_title("Price vs MA50 vs MA100", fontsize=10)
axes[1].legend()

axes[2].plot(ma_100_days, 'r', label='MA100')
axes[2].plot(ma_200_days, 'b', label='MA200')
axes[2].plot(data.Close, 'g', label='Stock Price')
axes[2].set_title("Price vs MA100 vs MA200", fontsize=10)
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

# Predictions
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Convert to scalar values
predicted_price = predict[-1].item()
actual_price = y[-1].item()

# Get the latest actual market price
latest_price = float(data['Close'].iloc[-1])
latest_date = data.index[-1]

# Calculate Market Change Percentage
market_change = ((predicted_price - actual_price) / actual_price) * 100

# Display market trend with color
trend_color = "#43a047" if market_change > 0 else "#e53935"
trend_icon = "📈" if market_change > 0 else "📉"
st.markdown(f"""
    <div style='text-align: center; padding: 10px; margin: 10px 0;'>
        <h3>{trend_icon} Market Trend</h3>
        <p style='font-size: 24px; color: {trend_color}; font-weight: bold;'>
            {market_change:+.2f}%
        </p>
        <p style='color: {trend_color};'>
            {"Bullish" if market_change > 0 else "Bearish"} Trend
        </p>
    </div>
""", unsafe_allow_html=True)

# List of major companies to track (US + Indian Companies)
companies = [
    # US Tech Companies
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSM', 'AVGO', 'ASML', 'AMD',
    # Indian Companies
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
    'WIPRO.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 'HCLTECH.NS', 'MARUTI.NS',
    # Additional Indian Companies
    'AXISBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'ULTRACEMCO.NS', 'ASIANPAINT.NS',
    'TITAN.NS', 'BAJAJFINSV.NS', 'SUNPHARMA.NS', 'ONGC.NS', 'NTPC.NS',
    'POWERGRID.NS', 'COALINDIA.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS',
    'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'
]
performance_data = []

# Update top performing companies section
for symbol in companies:
    try:
        stock_data = yf.Ticker(symbol)
        symbol_currency = get_currency_symbol(symbol)
        info = stock_data.info
        hist = stock_data.history(period='1y')
        
        # Calculate yearly performance
        yearly_change = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100
        
        # Calculate 30-day performance
        monthly_change = ((hist['Close'][-1] - hist['Close'][-30]) / hist['Close'][-30]) * 100
        
        performance_data.append({
            'Symbol': symbol,
            'Company': info.get('longName', 'N/A'),
            'Current Price': f"{symbol_currency}{hist['Close'][-1]:.2f}",
            '30-Day Change': f"{monthly_change:+.2f}%",
            'Yearly Change': f"{yearly_change:+.2f}%",
            'Market Cap (B)': f"{symbol_currency}{info.get('marketCap', 0)/1e9:.2f}B"
        })
    except:
        continue

# Create DataFrame and sort by yearly performance
df = pd.DataFrame(performance_data)

# Apply styling only if the DataFrame is not empty
if not df.empty:
    df_styled = df.style.apply(lambda x: ['background-color: #e6ffe6' if '+' in str(v) else 'background-color: #ffe6e6' 
                                        for v in x], subset=['30-Day Change', 'Yearly Change'])
    st.dataframe(df_styled, use_container_width=True)
else:
    try:
        # Calculate overall market performance
        if not data.empty and 'Close' in data.columns:
            current_price = data['Close'].iloc[-1]
            start_price = data['Close'].iloc[0]
            overall_change = ((current_price - start_price) / start_price) * 100
            
            st.warning(f"Market Overview for {stock}:\n" +
                      f"Starting Price: {currency_symbol}{start_price:.2f}\n" +
                      f"Current Price: {currency_symbol}{current_price:.2f}\n" +
                      f"Overall Change: {overall_change:+.2f}%")
        else:
            st.error(f"No data available for {stock}. Please check the stock symbol.")
    except Exception as e:
        st.error(f"Error processing data for {stock}. Please check the stock symbol.")

# Add performance insights
st.markdown("""
    <div style='font-size: 0.9em; margin-top: 15px;'>
        <h4>💡 Performance Insights:</h4>
        <ul>
            <li>🟢 Green background indicates positive performance</li>
            <li>🔴 Red background indicates negative performance</li>
            <li>📊 Data is updated in real-time</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Add Technical Analysis Section
st.markdown("""
    <div style='background: linear-gradient(45deg, #2937f0, #9f1ae2); 
                color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h2>📊 Technical Analysis</h2>
    </div>
""", unsafe_allow_html=True)

# Calculate RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
current_rsi = float(rsi.iloc[-1])  # Convert to float for comparison

# Calculate MACD
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
current_macd = float(macd.iloc[-1])  # Convert to float
current_signal = float(signal.iloc[-1])  # Convert to float

# Create Technical Indicators Display
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3>RSI (14)</h3>
            <p style="font-size: 24px; color: {};">{:.2f}</p>
            <p>{}</p>
        </div>
    """.format(
        "#e53935" if current_rsi > 70 else "#43a047" if current_rsi < 30 else "#1e88e5",
        current_rsi,
        "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3>MACD</h3>
            <p style="font-size: 24px; color: {};">{:.2f}</p>
            <p>{}</p>
        </div>
    """.format(
        "#43a047" if current_macd > current_signal else "#e53935",
        current_macd,
        "Bullish Crossover" if current_macd > current_signal else "Bearish Crossover"
    ), unsafe_allow_html=True)

with col3:
    # Calculate trend strength
    trend_strength = abs(market_change)
    st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3>Trend Strength</h3>
            <p style="font-size: 24px; color: {};">{:.2f}%</p>
            <p>{}</p>
        </div>
    """.format(
        "#1e88e5",
        trend_strength,
        "Strong" if trend_strength > 20 else "Moderate" if trend_strength > 10 else "Weak"
    ), unsafe_allow_html=True)

# Add Trading Signals
st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h3>💡 Trading Signals</h3>
        <ul style='list-style-type: none; padding: 0;'>
            <li style='margin: 10px 0; padding: 10px; border-radius: 5px; background-color: {};'>
                <strong>RSI Signal:</strong> {} (RSI: {:.2f})
            </li>
            <li style='margin: 10px 0; padding: 10px; border-radius: 5px; background-color: {};'>
                <strong>MACD Signal:</strong> {}
            </li>
            <li style='margin: 10px 0; padding: 10px; border-radius: 5px; background-color: {};'>
                <strong>Trend Signal:</strong> {} with {} strength
            </li>
        </ul>
    </div>
""".format(
    "#e6ffe6" if 30 < current_rsi < 70 else "#ffe6e6",
    "Neutral" if 30 < current_rsi < 70 else "Oversold" if current_rsi <= 30 else "Overbought",
    current_rsi,
    "#e6ffe6" if current_macd > current_signal else "#ffe6e6",
    "Bullish" if current_macd > current_signal else "Bearish",
    "#e6ffe6" if market_change > 0 else "#ffe6e6",
    "Bullish" if market_change > 0 else "Bearish",
    "strong" if trend_strength > 20 else "moderate" if trend_strength > 10 else "weak"
), unsafe_allow_html=True)
