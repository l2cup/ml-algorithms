%tensorflow_version 1.x

!pip install nltk
import nltk
nltk.download()

import math
import operator
import numpy as np
import csv
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation


# - - - - - - - - - - - - - - LOAD AND CLEANUP - - - - - - - - - - - - - - - - -

# Loads the data from the .csv file and extracts the needed classes and tweets
def extract_data(csvpath):
  print("Loading klasses and words...")
  klasses = []
  tweets = []
  with open(csvpath, 'r', encoding='latin1') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader, None)
    for row in reader:
      klasses.append(int(row[1]))
      tweets.append(row[2])
    print("Done loading.")
    return klasses, tweets

# Cleans up tweets and repacks them into a vector of words (each tweet)
def cleanup_tweets(tweets):
  print("Cleaning up tweets...")
  clean_tweets = []
  porter = PorterStemmer()
  stop_punc = set(stopwords.words('english')).union(set(punctuation)).union({'br'})
  for tweet in tweets:
    table = str.maketrans('', '', punctuation)
    stripped_tweet = ' '.join(tweet.split())
    words = word_tokenize(stripped_tweet)
    words_low = [w.lower() for w in words]
    words_filter = [w for w in words_low if w not in stop_punc and w.isalpha()]
    words_strip = [w.translate(table) for w in words_filter]
    words_stemm = [porter.stem(w) for w in words_filter]

    clean_tweets.append(words_stemm)

  print("Done cleaning.")
  return clean_tweets

# - - - -


# Creation of the vocabulary, skipping words with the length of 1 such as 'u'
def create_vocabulary(tweets):
  print("Creating vocabulary...")
  vocab_dictionary = dict()
  for tweet in tweets:
    for word in tweet:
      if len(word) > 1:
        vocab_dictionary.setdefault(word, 0)
        vocab_dictionary[word] += 1

  print("Done creating.")
  return sorted(vocab_dictionary, key=vocab_dictionary.get, reverse=True)[:10000]

def numocc_score(word, tweet):
  return tweet.count(word)

# Creates one row for our feature matrix
def create_bow_row(tweet, vocabulary):
  bow_row = np.zeros(len(vocabulary), dtype=np.float64)
  for cnt, word in enumerate(vocabulary):
    bow_row[cnt] = numocc_score(word, tweet)
  return bow_row

# Creates the feature matrix from loaded data that will go on into training
def create_feature_matrix(klasses, tweets, vocabulary):
  print("Creating features...")
  X = np.zeros((len(tweets), len(vocabulary)), dtype=np.float64)
  Y = np.zeros(len(tweets), dtype=np.int32)

  for cnt, tweet in enumerate(tweets):
    X[cnt] = create_bow_row(tweet, vocabulary)

  for klass_index in range(len(tweets)):
    Y[klass_index] = klasses[klass_index]

  print("Done creating.")
  return X, Y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - MODEL - - - - - - - - - - - - - - - - - - -

class Bayes:
  def __init__(self, nb_classes, nb_words, alpha):
    self.nb_classes = nb_classes
    self.nb_words   = nb_words
    self.alpha      = alpha

  def fit(self, X, Y):
    print("Training...")
    nb_examples = X.shape[0]

    self.priors = np.bincount(Y) / nb_examples

    occs = np.zeros((self.nb_classes, self.nb_words))
    for klass_cnt, Yi in enumerate(Y):
      klass = Yi
      for word_cnt, Xi in enumerate(X[klass_cnt]):
        occs[klass][word_cnt] += Xi

    self.likelihood = np.zeros((self.nb_classes, self.nb_words))
    for klass_index in range(self.nb_classes):
      for word_index in range(self.nb_words):
        numerator = occs[klass_index][word_index] + self.alpha
        denominator = np.sum(occs[klass_index]) + self.nb_words * self.alpha
        self.likelihood[klass_index][word_index] = numerator / denominator

    print("Done.")

  def predict(self, bow_row):
    probabilities = np.zeros(self.nb_classes)

    for klass_index in range(self.nb_classes):
      klass_prob = self.priors[klass_index]
      for word_index in range(self.nb_words):
        t = bow_row[word_index]
        klass_prob += t * np.log(self.likelihood[klass_index][word_index])

      probabilities[klass_index] = klass_prob

    prediction = np.argmax(probabilities)
    return prediction

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - c) - - - - - - - - - - - - - - - - - - - -

# Returns either top n positive or negative words used in the tweets
def get_top_n_words(tweets, klasses, wanted_klass, n):
  dictionary = dict()

  for cnt in range(len(klasses)):
    if klasses[cnt] == wanted_klass:
      for word in tweets[cnt]:
        dictionary.setdefault(word, 0)
        dictionary[word] += 1

  return sorted(dictionary, key=dictionary.get, reverse=True)[:n]

# Custom calculation for the ratio between the number of appearances in
# positive over negative tweets of one word
def LR(tweets, klasses, word):
  positives = 0
  negatives = 0
  for cnt, tweet in enumerate(tweets):
    if klasses[cnt] == 1:
      positives += tweet.count(word)
    else:
      negatives += tweet.count(word)

  if positives > 9 and negatives > 9:
    return (positives / negatives)
  else:
    return 0.0



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - EXECUTION - - - - - - - - - - - - - - - - - -

csvpath = '/content/drive/My Drive/ML/D1/4./twitter.csv'

klasses, tweets = extract_data(csvpath)
tweets = cleanup_tweets(tweets)

ratio_limit = math.floor(len(tweets) * 0.8)
training_tweets   = tweets[:ratio_limit]
training_klasses  = klasses[:ratio_limit]

test_tweets       = tweets[ratio_limit:]
test_klasses      = klasses[ratio_limit:]

vocabulary = create_vocabulary(tweets)

X, Y = create_feature_matrix(training_klasses, training_tweets, vocabulary)

model = Bayes(nb_classes=2, nb_words=10000, alpha=10)
model.fit(X, Y)

correct_predictions = 0

print("Predicting...")

TP = 0
FP = 0
TN = 0
FN = 0

for cnt, tweet in enumerate(test_tweets):
  bow_row = create_bow_row(tweet, vocabulary)

  prediction = model.predict(bow_row)

  if prediction == test_klasses[cnt]:
    correct_predictions += 1

  if prediction == 1:
    if test_klasses[cnt] == 1:
      TP += 1
    else:
      FP += 1

  else:
    if test_klasses[cnt] == 1:
      FN += 1
    else:
      TN += 1

confusion_matrix = [[TN, FP],
                    [FN, TP]]
accuracy = correct_predictions / len(test_tweets)
print('Accuracy:', accuracy, '%')
print('Confusion matrix: [[TN, FP],[FN, TP]]')
print(confusion_matrix)

recall = TP / (TP + FN)
precision = TP / (TP + FP)

F_measure = (2 * recall * precision) / (recall + precision)

print("CM Accuracy: ", F_measure, '%')

print()
print('Top 5 words in negative tweets:', get_top_n_words(tweets, klasses, 0, 5))
print('Top 5 words in positive tweets:', get_top_n_words(tweets, klasses, 1, 5))


LR_dictionary = dict()

for word in vocabulary:
  LR_dictionary.setdefault(word, 0.0)
  LR_dictionary[word] = LR(tweets, klasses, word)

print()
print('LRTOP:', sorted(LR_dictionary, key=LR_dictionary.get, reverse=True)[:5])
print('LRNOT:', sorted(LR_dictionary, key=LR_dictionary.get, reverse=False)[:5])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  # Top 5 negative
# Reci kao sto su 'go' i 'like' po nekoj pretpostavci bi mogli da se pojavljuju
# u oba tipa tweet-a, s tim sto je program za 'go' ocigledno pronasao
# negativnu asocijaciju, dok se 'like' zaista pojavluje u oba tipa tweet-a
# 'miss' je ocigledno negativna konotacija pa se zato i pojavluje u prvih pet

  # Top 5 positive
# 'thank' bi se najverovatnije tumacila kao rec vezana za zahvalnost pa samim
# tim spada u pozitivne tweet-ove. Reci 'good' i 'love' se ocigledno koriste
# u pozitivnom kontekstu. 'good' kao opis, pohvala ili poredjenje, a 'love'
# kao stanje, sto je jelte uglavnom pozitivno.

  # Top 5 LR best
# Reci 'welcom', 'appreci', 'congrat' (nakon jezivog stemovanja) se u bukvalnom
# smislu mogu shvatiti kao reci dobrodoslice, pohvale i zahvalnosti..
# doduse, u nekom skrivenom ili sarkasticnom kontekstu takodje bi sve tri reci
# imale prolaznost, pa samim tim pojavu i u negativnim tweet-ovima.

  # Top 5 LR worst
# Reci 'cancel' i 'bummer' jasno opisuju neko negativno stanje, takodje u
# bukvalnom smislu. Za rec 'bummer' vazi isto kao i za reci sa visokom metrikom,
# u skrivenom ili prenesenom znacenju moze da ima pozitivan smisao.

  # LR metrika
# Metrika koju smo koristili nam opisuje odnos pojave reci u pozitivnim i
# negativnim tweet-ovima. 
