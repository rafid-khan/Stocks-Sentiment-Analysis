'''
File: main.py 
This application prompts the user for a stock ticker name, which is then
used to search FinViz, a browser-based stock market research platform,
for the latest news articles related to the stock. The website is then stored
and parsed for the articles' headlines and time. Scores are then assigned to
each headline based on sentiment analyis. These scores are then displayed to the
user in the context of both hourly, and daily headlines. 
'''

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly
import plotly.express as px
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import flask
from flask import Flask, render_template

__author__ = 'Rafid Khan'
__copyright__ = None
__credits__ = 'Rafid Khan'

__maintainer__ = 'Rafid Khan'
__email__ = 'rafid3075@gmail.com'
__status__ = 'Production'


def get_news(ticker):
    '''
    This function takes in a stock ticker and then sends an HTTP request of
    the url containing the latest headlines related to that particular 
    stock found on FinViz. The page element containing the headlines, and 
    time is then returned using BeautifulSoup.  
    
    :param ticker: Stock ticker that will be used to query FinViz site for 
    relevant news 
    :return news_table: News-table element found on the accessed website 
    (raw HTML code)
    '''
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'lxml')
    news_table = html.find(id='news-table')

    return news_table


def parse_news(news_table):
    '''
    Takes in raw HTML code (news_table) and uses the BeautifulSoup library to 
    parse headlines and corresponding dates/times into a Pandas DataFrame

    :param news_table: News-table element found on the accessed website 
    (raw HTML code)
    :return parsed_news_df: Pandas DataFrame containing parsed headlines and 
    corresponding dates & time
    '''

    parsed_news = []
    for x in news_table.findAll('tr'):
        try:
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            parsed_news.append([date, time, text])
            columns = ['date', 'time', 'headline']
            parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
            parsed_news_df['datetime'] = pd.to_datetime \
                (parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        except AttributeError:
            continue

    return parsed_news_df


def score_news(parsed_news_df):
    '''
    Takes in a DataFrame containing parsed headlines and assigns a sentiment
    score to each headline using the NLTK Vader library. The sentiment scores
    are appended to the DataFrame as additional columns. 

    :param parsed_news_df: Pandas DataFrame containing parsed headlines and 
    corresponding dates & time
    :return parsed_and_scored_news: Pandas DataFrame containing parsed 
    headlines + date/time as well as sentiment scores
    '''
    vader = SentimentIntensityAnalyzer()
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)
    parsed_and_scored_news = parsed_and_scored_news.rename\
        (columns={"compound": "sentiment_score"})

    return parsed_and_scored_news


def plot_hourly_sentiment(parsed_and_scored_news, ticker):
    '''
    Resamples the hourly sentiment scores found in the inputted DataFrame
    and charts results using the Plotly package

    :param parsed_and_scored_news: Pandas DataFrame containing parsed 
    headlines + date/time as well as sentiment scores
    :param ticker: Stock ticker that will be used to query FinViz site for 
    relevant news 
    :return fig: Figure illustrating resampled hourly sentiment scores for
    ticker
    '''
    mean_scores = parsed_and_scored_news.resample('H').mean()
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', \
        title=ticker + ' Hourly Sentiment Scores')
    return fig


def plot_daily_sentiment(parsed_and_scored_news, ticker):
    '''
    Resamples the daily sentiment scores found in the inputted DataFrame
    and charts results using the Plotly package

    :param parsed_and_scored_news: Pandas DataFrame containing parsed 
    headlines + date/time as well as sentiment scores
    :param ticker: Stock ticker that will be used to query FinViz site for 
    relevant news 
    :return fig: Figure illustrating resampled daily sentiment scores for
    ticker
    '''
    mean_scores = parsed_and_scored_news.resample('D').mean()
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', \
        title=ticker + ' Daily Sentiment Scores')
    return fig


app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sentiment', methods=['POST'])
def sentiment():
    ticker = flask.request.form['ticker'].upper()
    news_table = get_news(ticker)
    parsed_news_df = parse_news(news_table)
    parsed_and_scored_news = score_news(parsed_news_df)
    fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
    fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)

    graphJSON_hourly = json.dumps(fig_hourly, \
        cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_daily = json.dumps(fig_daily, \
        cls=plotly.utils.PlotlyJSONEncoder)

    header = "Hourly and Daily Sentiment of {} Stock".format(ticker)
    description = """
	The above chart averages the sentiment scores of {} stock hourly \
    and daily.The table below gives each of the most recent \
    headlines of the stock and the negative, neutral, positive and an \
    aggregated sentiment score. The news headlines are obtained from \
    the FinViz website. Sentiments are given by the \
    nltk.sentiment.vader Python library.
    """.format(ticker)
    return render_template('sentiment.html', \
        graphJSON_hourly=graphJSON_hourly, graphJSON_daily=graphJSON_daily,
        header=header, table=parsed_and_scored_news.to_html(classes='data'), \
        description=description)


if __name__ == '__main__':
    app.debug = True
    app.run()

    
