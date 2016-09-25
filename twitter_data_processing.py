import pandas as pd 
import pprint 
import string
import operator
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk import *
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
      tfidf_vectorizer - tfidf vectorizer which is fitted on the input data, and can be applied to new tweets for preprocessing
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
    positive_words = set(['wow', 'beautiful', 'amazing', 'won', 'want', 'really cool', 'feel better', 'good'])    # Constructing a set of postive words from tweet messages. 
    dense = self.tfidf_vectorizer.transform([tweet]).toarray()[0]                                                 # Find the tokens of tweet which are part of vocabulary 
    dense = np.where(dense > 0)[0]                                              
    terms = set([self.tfidf_vectorizer.get_feature_names()[x] for x in dense])                                    # Converting the index list to actual feature names
    return len(terms.intersection(positive_words))/(len(terms) + 1.0)                                             # Adding 1 in denominator to prevent division by 0. 

  def negative_word(self, tweet):
    """
    This feature computes the proportion of negative words in a given tweet.
    """
    negative_words = set(['wrong', 'worst', 'warned', 'dont like', 'upset', 'ugh', 'bad'])                        # Using the tweet data to find negative words
    dense = self.tfidf_vectorizer.transform([tweet]).toarray()[0]
    dense = np.where(dense > 0)[0]
    terms = set([self.tfidf_vectorizer.get_feature_names()[x] for x in dense])
    return len(terms.intersection(negative_words))/(len(terms) + 1.0)

  def pos_features(self, tweet):
    """
    This feature computes the number of noun, pronoun, adjective, adverb and other parts of speech for an input tweet. 
    Input 
         Tweet Message 
    Output - 
         dict_ 
             NN  - Nouns 
             VBP - Verbs
             PRP - Pronoun 
             RB  - Adverb 
             JJ  - Adjective 
    """
    text = word_tokenize(tweet)                                                                  # Tokenize the tweet message
    pos_tagged = nltk.pos_tag(text)                                                              # Find POS tags from NLTK package 
    pos_feature = {'NN': 0, 'VBP': 0, 'PRP': 0, 'RB': 0, 'JJ': 0}
    for elem in pos_tagged:                                                                      # Find the count of each pos tag 
      if elem[1] in pos_feature:
        pos_feature[elem[1]] += 1
      else:
        pos_feature[elem[1]] = 1
    return pos_feature
    
  def read_data(self):
    """
    Read the Input file and extract the tweets. 
    Do some preprocessing to the input tweet data. 
    """
    with open(self.filename, "r") as f:                                                           # Read the input file. 
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
    print self.tweets[:5]

    # Convert the tweets to bag of words representation
    self.countVectorizer = CountVectorizer(decode_error='ignore', \
                                           stop_words='english', \
                                           min_df=5, ngram_range=(1, 2))                         # Extract uni-gram, bi-grams and tri-grams from the tweets 
    self.bag_of_words = self.countVectorizer.fit_transform(self.tweets)                          # Convert each tweet into a vector 

    print "\nTop 20 uni grams in the vocabulary_"
    print sorted(dict((key,value) for key, value in self.countVectorizer.vocabulary_.iteritems() if key.count(' ')==0).items(), key=operator.itemgetter(1), reverse=True)[:20]

    print "\nTop 20 bi-grams in vocbulary "
    print sorted(dict((key,value) for key, value in self.countVectorizer.vocabulary_.iteritems() if key.count(' ')==1).items(), key=operator.itemgetter(1), reverse=True)[:20]

    # Convert the Tweets to TF - IDF Representation for understanding importance of individual words  
    self.tfidf_vectorizer = TfidfVectorizer(decode_error='ignore',\
                                             stop_words='english', \
                                             min_df=10, ngram_range=(1, 3))                      # Convert the tweets message to tf idf representation 
    self.tf_idf_scores = self.tfidf_vectorizer.fit_transform(self.tweets)                        
  
    # Convert the tf - idf to pandas dataframe 
    print "\nTf - Idf for each tweet in the dataset"
    self.df = pd.DataFrame(self.tf_idf_scores.toarray(), columns=self.tfidf_vectorizer.get_feature_names())   # Convert the td idf values for each tweet into a DataFrame
    self.df["Input Tweets"] = self.tweets
    print self.df.sample(n=5)               

    # Adding Proportion of positive words as a feature
    self.df['Positive Words'] = map(lambda x: self.positive_word(x), self.df['Input Tweets'])    # Adding proportion of positive words as a feature
    print self.df.sample(n=5)
   
    # Adding Proportion of negative words as a feature 
    self.df['Negative Words'] = map(lambda x: self.negative_word(x), self.df['Input Tweets'])    # Adding proportion of negative words as a feature
    print self.df.sample(n=5)

    # Adding part of speech tag features to the dataframe 
    pos_feat_ = map(lambda x: self.pos_features(x), self.df['Input Tweets'])                     # Adding number of parts of speech like Noun, Pronoun, Adjective as a feature
    self.df['Nouns'] = map(lambda x: x['NN'], pos_feat_)
    self.df['Verbs'] = map(lambda x: x['VBP'], pos_feat_)
    self.df['Pronoun'] = map(lambda x: x['PRP'], pos_feat_)
    self.df['Adverb'] = map(lambda x: x['RB'], pos_feat_)
    self.df['Adjective'] = map(lambda x: x['JJ'], pos_feat_)
    print self.df.sample(n=5)

if __name__ == "__main__":
  preprocess = Preprocess("sts_gold_tweet.csv")
  preprocess.preprocess_data()
