
import csv
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
import re
from twitter import *


twitter_api = Twitter(auth=OAuth(token='YOUR_TWITTER_TOKEN', token_secret='YOUR_TWITTER_TOKEN_SECRET',
                                 consumer_key='YOUR_API_KEY', consumer_secret='YOUR_API_SECRET'))

def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.search.tweets(
            q=search_keyword, count=100, tweet_mode="extended")

        return [{"full_text": status['full_text'], "label":None} for status in tweets_fetched['statuses']]

    except:
        print("Unfortunately, something went wrong..")
        return None


search_term = input("Enter company/organization name to search twitter: ")
testDataSet = buildTestSet(search_term + " -filter:retweets")


def buildTrainingSet(disasterTweets):
    trainingDataSet = []

    with open(disasterTweets, 'r') as trainingDataCSV:
        linereader = csv.reader(trainingDataCSV, delimiter=',', quotechar="\"")

        for row in linereader:
            trainingDataSet.append(
                {"unit_id": row[0], "trusted_judgments": row[1], "confidence": row[2], "text": row[3], })

    return trainingDataSet


trainingData = buildTrainingSet("./userTweets.csv")


class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words(
            'english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTweet(self, tweet):
        processedTweet = self._processTweet(tweet['text'])

        return processedTweet

    def processTweets(self, list_of_tweets):
        processedTweets = []
        for tweet in list_of_tweets:
            processedTweets.append(
                (self._processTweet(tweet["text"]), tweet["confidence"]))
        return processedTweets

    def _processTweet(self, tweet):
        tweet = tweet.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',
                       'URL', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        # remove repeated characters (helloooooooo into hello)
        tweet = word_tokenize(tweet)
        return [word for word in tweet if word not in self._stopwords]


tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)


def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures = nltk.classify.apply_features(
    extract_features, preprocessedTrainingSet)

NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
NBResultLabels = [NBayesClassifier.classify(
    extract_features(tweet[0])) for tweet in preprocessedTestSet]

if NBResultLabels.count('1') > NBResultLabels.count('<=0.5'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " +
          str(100*NBResultLabels.count('1')/len(NBResultLabels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " +
          str(100*NBResultLabels.count('<=0.5')/len(NBResultLabels)) + "%")
