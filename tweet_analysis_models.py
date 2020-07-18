# utilities
import re
import time
import pickle
import numpy as np
import pandas as pd

# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from yellowbrick.text import FreqDistVisualizer
from yellowbrick import set_palette

# nltk
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('./training.1600000.processed.noemoticon.csv',
                      encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Removing the unnecessary columns.
dataset = dataset[['sentiment', 'text']]
# Replacing the values to ease understanding.
dataset['sentiment'] = dataset['sentiment'].replace(4, 1)

# Plotting the distribution for data set.
ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                               legend=False)
ax.set_xticklabels(['Negative', 'Positive'], rotation=0)

plt.show()

# Storing data in lists.
text, sentiment = list(dataset['text']), list(dataset['sentiment'])

# Defining dictionary containing all emojis with their meanings.
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
stop_word_list = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
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

# Defining set of negations
appostrophes = {"aren't": "are not",
                "can't": "cannot",
                "couldn't": "could not",
                "won't": "would not",
                "shouldn't": "should not",
                "hadn't": "had not",
                "hasn't": "has not",
                "haven't": "have not",
                "mustn't": "must not",
                "isn't": "is not",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not"}


def preprocess(textdata):
    processed_text = []

    # Create Lemmatizer.
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
        tweet = re.sub(url_pattern, '', tweet)
        # Replace negation
        for appos in appostrophes.keys():
            tweet = tweet.replace(appos, appostrophes[appos])
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, emojis[emoji])
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(user_pattern, '', tweet)
        # Replace all non alphabets.
        tweet = re.sub(alpha_pattern, ' ', tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            if word not in stop_word_list:
                if len(word) > 1:
                    # Lemmatizing the word.
                    word = word_lemm.lemmatize(word)
                    tweetwords += (word + ' ')

        processed_text.append(tweetwords)

    return processed_text


t = time.time()
processed_text = preprocess(text)
print("Text Preprocessing complete.")
print('Time Taken: ', round(time.time() - t), 'seconds')
print()

# word clouds for negative tweets
data_neg = processed_text[:800000]
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(data_neg))
plt.axis('off')
plt.imshow(wc)
plt.show()

# word clouds for positive tweets
data_pos = processed_text[800000:]
plt.figure(figsize=(20, 20))
wc2 = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(data_pos))
plt.axis('off')
plt.imshow(wc2)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(processed_text, sentiment, test_size=0.05, random_state=0)
print("Data Split done.")
print()

t = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
vectorizer.fit(X_train)
print("Vectoriser fitted")
print("No. of feature_words: ", len(vectorizer.get_feature_names()))
print('Time Taken: ', round(time.time() - t), 'seconds')

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
print("Data transformed")
print()

set_palette('sns_pastel')
visualizer = FreqDistVisualizer(
    features=vectorizer.get_feature_names()
)
visualizer.fit(X_train)
visualizer.show()


# evaluate model
def model_evaluate(model):
    # predict values for text data-set
    y_pred = model.predict(X_test)

    # print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # compute and plot the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='', xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()


t = time.time()
print("Logistic Regression")
LR_model = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
LR_model.fit(X_train, y_train)
model_evaluate(LR_model)
print('Time Taken: ', round(time.time() - t), 'seconds')

t = time.time()
print("Bernoulli Naive Bayes")
BNB_model = BernoulliNB(alpha=2)
BNB_model.fit(X_train, y_train)
model_evaluate(BNB_model)
print('Time Taken: ', round(time.time() - t), 'seconds')

t = time.time()
print("Linear SVC")
SVC_model = LinearSVC()
SVC_model.fit(X_train, y_train)
model_evaluate(SVC_model)
print('Time Taken: ', round(time.time() - t), 'seconds')

file = open('vectorizer-ngram-(1,2).pickle', 'wb')
pickle.dump(vectorizer, file)
file.close()

file = open('Sentiment-BNB.pickle', 'wb')
pickle.dump(BNB_model, file)
file.close()

file = open('Sentiment-SVC.pickle', 'wb')
pickle.dump(SVC_model, file)
file.close()

file = open('Sentiment-LR.pickle', 'wb')
pickle.dump(LR_model, file)
file.close()
