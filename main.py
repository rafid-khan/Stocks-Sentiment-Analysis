import json
from tokenize import String
from urllib import response
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly
import plotly.express as px
import json
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

finviz_url = 'https://finviz.com/quote.ashx?t='

def get_news(ticker: String):
    url = finviz_url + ticker
    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
    response = urlopen(req)
    html = BeautifulSoup(response)
    news_table = html.find(id='news-table')
    return news_table

def parse_news(news_table):
    parsed_news = []
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]
        else:
            date = date_scrape[0]
            time = date_scrape[1]

        parse_news.append([date, time, text])
        columns = ['date', 'time', 'headline']
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])

    return parsed_news_df

def score_news(parsed_news_df):
    vader = SentimentIntensityAnalyzer()
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_news_df.set_index('datetime')
    parsed_and_scored_news = parsed_news_df.drop(['date', 'time'], 1)
    parsed_and_scored_news = parsed_news_df.rename(columns={"compound": "sentiment_score"})
    return parsed_and_scored_news

def plot_hourly_sentiment(parsed_and_scored_news, ticker):
    mean_scores = parsed_and_scored_news.resample('H').mean()
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_scores', title=ticker + ' Hourly Sentiment Scores')
    return fig

def plot_daily_sentiment(parsed_and_scored_news, ticker):
    mean_scores = parsed_and_scored_news.resample('D').mean()
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title=ticker + ' Daily Sentiment Scores')
    return fig




