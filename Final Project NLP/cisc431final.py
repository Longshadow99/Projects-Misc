import os
import nltk
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import re, string
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.metrics.scores import recall
from nltk.metrics.scores import precision
from nltk.metrics.scores import f_measure
import collections


corpusdir = 'C:/Users/adamh/Desktop/Final Project NLP/corpus' # Directory of corpus.

newcorpus = PlaintextCorpusReader(corpusdir, '.*')

neg_sents = newcorpus.sents(newcorpus.fileids()[0])
pos_sents = newcorpus.sents(newcorpus.fileids()[1])
stop_words = stopwords.words('english')

neg_words = newcorpus.words(newcorpus.fileids()[0])
pos_words = newcorpus.words(newcorpus.fileids()[1])

print("Avg Sent Legnth: ",(sum(len(sent) for sent in neg_sents) / len(neg_sents) + sum(len(sent) for sent in pos_sents) / len(pos_sents))/2)
print("Avg Word Legnth: ",(sum(len(word) for word in neg_words) / len(neg_words) + sum(len(word) for word in pos_words) / len(pos_words))/2)

# Using modified code from this tutorial to help me clean my tokens easierhttps://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
def remove_noise(tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tokens):

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
clean_neg = []
for x in neg_sents:
	clean_neg.append(remove_noise(x,stop_words))
clean_pos = []
for x in pos_sents:
	clean_pos.append(remove_noise(x,stop_words))



fdist1 = FreqDist()
fdist2 = FreqDist()
for sentance in clean_neg:
	for word in sentance:
		fdist1[word] += 1
for sentance in clean_pos:
	for word in sentance:
		fdist2[word] += 1
fdist1.pprint(10)
fdist2.pprint(10)

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

fix_neg = get_tweets_for_model(clean_neg)
fix_pos = get_tweets_for_model(clean_pos)



neg = [(tweet_dict, "neg")
                     for tweet_dict in fix_neg]

pos = [(tweet_dict, "pos")
                     for tweet_dict in fix_pos]

labeled = neg + pos
random.shuffle(labeled)
train = labeled[:840]
test = labeled[840:]

classifier = NaiveBayesClassifier.train(train)

#Using a modified version of this code to calculate precision, recall and f measure easier https://streamhacker.com/2010/05/17/text-classification-sentiment-analysis-precision-recall/
print("Accuracy:", classify.accuracy(classifier, test))
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(test):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))
print(classifier.show_most_informative_features(10))
