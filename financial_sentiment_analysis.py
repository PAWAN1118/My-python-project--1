import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import yfinance as yf
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

news_data = [
    {"text": "Tech stock prices surged after a strong earnings report from Apple", "label": "positive"},
    {"text": "Investors are cautious as market volatility rises due to economic uncertainty", "label": "negative"},
    {"text": "The market remains flat with no significant changes in the major indices", "label": "neutral"},
]

df = pd.DataFrame(news_data)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['predicted_sentiment'] = df['cleaned_text'].apply(get_sentiment)

print(classification_report(df['label'], df['predicted_sentiment']))

df['predicted_sentiment'].value_counts().plot(kind='bar', title="Sentiment Distribution")
plt.show()

stock_symbol = 'AAPL'
stock_data = yf.download(stock_symbol, period="6mo", interval="1d")

print(stock_data.tail())

plt.plot(stock_data['Close'])
plt.title(f'{stock_symbol} Stock Price (Last 6 Months)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

df['sentiment_score'] = df['predicted_sentiment'].apply(lambda x: 1 if x == 'positive' else (-1 if x == 'negative' else 0))

sentiment_df = df[['text', 'sentiment_score']].copy()
sentiment_df['date'] = pd.to_datetime(['2023-11-01', '2023-11-02', '2023-11-03'])
sentiment_df.set_index('date', inplace=True)

stock_data_resampled = stock_data.resample('D').last()
stock_data_resampled = stock_data_resampled.reset_index()

combined_df = stock_data_resampled.join(sentiment_df, on='Date', how='left')
combined_df.set_index('Date', inplace=True)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price', color='tab:blue')
ax1.plot(combined_df.index, combined_df['Close'], color='tab:blue', label='Stock Price')

ax2 = ax1.twinx()
ax2.set_ylabel('Sentiment Score', color='tab:orange')
ax2.plot(combined_df.index, combined_df['sentiment_score'], color='tab:orange', label='Sentiment Score', linestyle='--')

plt.title(f'{stock_symbol} Stock Price and Sentiment Correlation')
plt.show()

stock_data['price_change'] = stock_data['Close'].pct_change()
stock_data['sentiment_score'] = combined_df['sentiment_score']

stock_data.dropna(inplace=True)

X = stock_data[['sentiment_score']]
y = np.where(stock_data['price_change'] > 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
