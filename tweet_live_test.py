from tweepy import API
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import tweepy as tw
import twitter_credentials

import pickle
import re

import pandas as pd
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer


# # # # TWITTER CLIENT # # # #
class TwitterClient:
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client


# # # # TWITTER AUTHENTICATOR # # # #
class TwitterAuthenticator:

    @staticmethod
    def authenticate_twitter_app():
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', ':-(': 'sad',
          ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked',
          ':-$': 'confused', ':\\': 'annoyed', ':#': 'mute', ':X': 'mute', ':^)': 'smile',
          ':-&': 'confused', '$_$': 'greedy', '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile',
          ':-0': 'yell', 'O.o': 'confused', '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile',
          ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip',
          '=^.^=': 'cat', '😃': 'smile', '😄': 'smile', '😁': 'grin', '😆': 'smile', '😅': 'shy',
          '😉': 'wink', '😊': 'blush', '☺': 'blush', '🐱': 'cat', '😂': 'laugh', '😚': 'kiss',
          '😙': 'kiss', '😍': 'love', '😐': 'neutral', '😑': 'expressionless', '😏': 'smirk',
          '☹': 'sad', '🥺': 'sad', '😢': 'cry', '😭': 'cry', '😩': 'weary', '😡': 'angry',
          '😠': 'angry', '😲': 'astonished', '😧': 'anguished', '😨': 'fear', '😰': 'anxious',
          '😵': 'dizzy', '😴': 'sleep', '😌': 'relieved'}
          
# Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']


# # # # TWITTER STREAM LISTENER # # # #
# noinspection PyCompatibility
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename):
        super().__init__()
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer:
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    @staticmethod
    def clean_tweet(textdata):
        processed_text = []

        # Create Lemmatizer and Stemmer.
        word_lemm = WordNetLemmatizer()

        # Defining regex patterns.
        url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        user_pattern = '@[^\s]+'
        alpha_pattern = "[^a-zA-Z0-9]"
        sequence_pattern = r"(.)\1\1+"
        seq_replace_pattern = r"\1\1"

        for tweet in textdata:
            tweet = tweet.lower()

            # Replace all URls with 'URL'
            tweet = re.sub(url_pattern, ' URL', tweet)
            # Replace all emojis.
            for emoji in emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
                # Replace @USERNAME to 'USER'.
            tweet = re.sub(user_pattern, ' USER', tweet)
            # Replace all non alphabets.
            tweet = re.sub(alpha_pattern, " ", tweet)
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

            tweetwords = ''
            for word in tweet.split():
                # Checking if the word is a stopword.
                # if word not in stopwordlist:
                if len(word) > 1:
                    # Lemmatizing the word.
                    word = word_lemm.lemmatize(word)
                    tweetwords += (word + ' ')

            processed_text.append(tweetwords)

        return processed_text

    @staticmethod
    def tweets_to_data_frame(tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])
        return df

    @staticmethod
    def load_models():
        # Load the vectorizer
        file = open('./vectorizer-ngram-(1,2).pickle', 'rb')
        vectorizer = pickle.load(file)
        file.close()
        # Load the LR model
        file = open('./Sentiment-LR.pickle', 'rb')
        LR_model = pickle.load(file)
        file.close()

        return vectorizer, LR_model

    def predict(self, vectoriser, model, text):
        # Predict the sentiment
        textdata = vectoriser.transform(self.clean_tweet(text))
        sentiment = model.predict(textdata)

        # Make a list of text with sentiment.
        data = []
        for text, pred in zip(text, sentiment):
            data.append((text, pred))

        # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(data, columns=['text', 'sentiment'])
        df = df.replace([0, 1], ["Negative", "Positive"])
        return df


if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_twitter_client_api()
    # Loading the models.
    vectoriser, LR_model = tweet_analyzer.load_models()

    search_words = input("Enter keyword: ")

    tweets = tw.Cursor(api.search, q=search_words, lang="en").items(50)

    df = tweet_analyzer.tweets_to_data_frame(tweets)

    tweet_sentiment = tweet_analyzer.predict(vectoriser, LR_model, df['tweets'].tolist())

    pos_count = 0
    neg_count = 0
    for sentiment in tweet_sentiment.sentiment:
        if sentiment == 'Positive': pos_count += 1
        if sentiment == 'Negative': neg_count += 1

    print(tweet_sentiment.head(20))

    labels = 'Positive', 'Negative'
    sizes = [pos_count, neg_count]
    colors = ['lightskyblue', 'gold']
    explode = (0.1, 0)

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()
