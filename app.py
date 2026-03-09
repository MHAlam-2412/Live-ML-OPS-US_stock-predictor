import streamlit as st
import yfinance as yf
import finnhub
import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import warnings

# 1. PAGE CONFIG MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide") 

# Ignore the version warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)

# --- SETUP ---
finnhub_client = finnhub.Client(api_key="d6ms4i9r01qir35i93hgd6ms4i9r01qir35i93i0")
analyzer = SentimentIntensityAnalyzer()

@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

model = load_model()

# --- BACKEND FUNCTIONS ---
def get_market_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    last_day = hist.iloc[-1]
    return {
        "Close": last_day['Close'],
        "Volume": last_day['Volume'],
        "High": last_day['High'],
        "Low": last_day['Low']
    }

def get_news_sentiment(ticker):
    today = datetime.today().strftime('%Y-%m-%d')
    news = finnhub_client.company_news(ticker, _from=today, to=today)
    if not news:
        return 0.0 
    scores = [analyzer.polarity_scores(n['headline'])['compound'] for n in news[:10]]
    return sum(scores) / len(scores)

# --- WEB INTERFACE ---

# Professional Sidebar
with st.sidebar:
    st.write("### Pipeline Architecture")
    st.write("🔹 **Algorithm:** Random Forest")
    st.write("🔹 **Market Data:** Live yfinance API")
    st.write("🔹 **NLP Engine:** VADER Sentiment")
    st.divider()
    st.caption("⚙️ Engineered by **Hamid Alam**")

st.title("📈 Live Stock Return Predictor")
st.write("An MLOps pipeline leveraging live market data and real-time NLP sentiment analysis.")

# User inputs
ticker = st.selectbox("Select an Asset", ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"])

# The big prediction button
if st.button("Generate Forecast"):
    with st.spinner(f"Fetching live market & sentiment data for {ticker}..."):
        market_data = get_market_data(ticker)
        sentiment = get_news_sentiment(ticker)
        
        # Build the exact dataframe your model needs
        input_data = {
            "Date": [datetime.today().strftime('%Y-%m-%d')], 
            "Ticker": [ticker],
            "Close": [market_data["Close"]],
            "Volume": [market_data["Volume"]],
            "High": [market_data["High"]],
            "Low": [market_data["Low"]],
            "News_Sentiment": [sentiment]
        }
        df = pd.DataFrame(input_data)
        
        # Generate Prediction
        prediction = model.predict(df)[0]
        
        # Calculate the predicted dollar price!
        predicted_price = market_data['Close'] * (1 + prediction)
        
        # Display the results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Live News Sentiment [-1 to 1]", value=f"{sentiment:.3f}")
            st.write(f"Last Close: ${market_data['Close']:.2f}")
            
        with col2:
            st.metric(
                label="Predicted Next-Day Close", 
                value=f"${predicted_price:.2f}", 
                delta=f"{(prediction * 100):.2f}%"
            )