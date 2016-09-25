import pandas as pd 
import pprint 
import string
import operator
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocess:
  
  def __init__(self, filename):
    """
    Parameters - 
      Filename - File containing the data for preprocessing 
      tweets - text messages containing the tweet data 
      bag_of_words - Matrix containing the bag of words representation of tweet messages
      countVectorizer - countVectorizer object which can be used to process new tweet messages. 
    """
    self.filename = filename
    self.tweets = []
    self.bag_of_words = None
    self.countVectorizer = None
    self.tfidf_vectorizer = None
    self.tf_idf_scores = None
    self.df = None

  def positive_word(self, tweet):
    """
    This feature computes the proportion of positive words in a given tweet. 
    """
    positive_words = set(['wow', 'beautiful', 'amazing', 'won', 'want', 'really cool', 'feel better', 'good'])
    dense = self.tfidf_vectorizer.transform([tweet]).toarray()[0]
    dense = np.where(dense > 0)[0]
    terms = set([self.tfidf_vectorizer.get_feature_names()[x] for x in dense])
    return len(terms.intersection(positive_words))/(len(terms) + 1.0)

  def read_data(self):
    """
    Read the Input file and extract the tweets. 
    Do some preprocessing to the input tweet data. 
    """
    with open(self.filename, "r") as f:                                                                # Read the input file. 
      self.tweets.append(f.read())                                                                # Append the tweets to tweets list. 
    f.close()                                      

  def stem_message(self, text):        
    """
    This function converts the each word of an input message to its base form. 
    """
    stemmer = SnowballStemmer("english")
    try:
      text = ''.join(stemmer.stem(word) for word in text)
      return text
    except:
      return text 

  def preprocess_data(self):
    """
    This function does the initial preprocessing of the data. 
    1) Split the data into individual tweets. 
    2) Remove the punctuation from the tweets and convert them to lower case. 
    3) Remove the stop words from the tweets. 
    4) Apply stemming to the tweets to convert them to there canonical form. 
    """
    # Read the Input data 
    self.read_data()

    # Initial preprocessing
    self.tweets = map(lambda x: x.split("\r\n"), self.tweets)                                     # Splitting the tweets into individual tweets
    self.tweets = map(lambda x: x.split(";")[2], self.tweets[0])                                  # Splitting each tweet with colon and extract the tweet message
 
    # Removing punctuation 
    exclude = set(string.punctuation)                                                             # Make a set of punctuation to remove it from tweet
    self.tweets = map(lambda x:''.join(ch for ch in x if ch not in exclude).lower(), self.tweets) # Remove the punctuation from each tweet 
    self.tweets = map(lambda x:' '.join(x.split()), self.tweets)                                  # Remove extra spaces 
    
    # Removing Stop Words 
    stopWords = set(stopwords.words("english"))                                                   # Removing Stop words 
    self.tweets = map(lambda x:' '.join(word for word in x.split() if word not in stopWords), self.tweets)

    # Applying Stemming 
    stemmer = PorterStemmer()
    self.tweets = map(lambda x: self.stem_message(x), self.tweets)                                # Convert each message to its base form after stemming 

    print "\nIntermediate Data After Initial Preprocessing"
    pprint.pprint(self.tweets[:5])

    # Convert the tweets to bag of words representation
    self.countVectorizer = CountVectorizer(decode_error='ignore', \
                                           stop_words='english', \
                                           min_df=5, ngram_range=(1, 2))                         # Extract uni-gram, bi-grams and tri-grams from the tweets 
    self.bag_of_words = self.countVectorizer.fit_transform(self.tweets)                          # Convert each tweet into a vector 

    print "\nTop 20 uni grams in the vocabulary_"
    print sorted(dict((key,value) for key, value in self.countVectorizer.vocabulary_.iteritems() if key.count(' ')==0).items(), key=operator.itemgetter(1), reverse=True)[:100]
    print "\nTop 20 bi-grams in vocbulary "
    print sorted(dict((key,value) for key, value in self.countVectorizer.vocabulary_.iteritems() if key.count(' ')==1).items(), key=operator.itemgetter(1), reverse=True)[:100]    
    # Convert the Tweets to TF - IDF Representation for understanding importance of individual words  
    self.tfidf_vectorizer = TfidfVectorizer(decode_error='ignore',\
                                             stop_words='english', \
                                             min_df=10, ngram_range=(1, 3)) 
    self.tf_idf_scores = self.tfidf_vectorizer.fit_transform(self.tweets)
  
    # Convert the tf - idf to pandas dataframe 
    print "\nTf - Idf for each tweet in the dataset"
    self.df = pd.DataFrame(self.tf_idf_scores.toarray(), columns=self.tfidf_vectorizer.get_feature_names())
    self.df["Input Tweets"] = self.tweets
    print self.df.sample(n=5)

    self.df['Positive Words'] = map(lambda x: self.positive_word(x), self.df['Input Tweets'])
    print self.df.sample(n=5)


if __name__ == "__main__":
  preprocess = Preprocess("sts_gold_tweet.csv")
  preprocess.preprocess_data()
