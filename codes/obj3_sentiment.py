'''
    NUS CS4248 Assignment 1 - Objective 3 (Regular Expression, Sentiment Analysis)

    Class Tokenizer for handling Objective 3

    Important: please strictly comply with the input/output formats for
               the methods of process_text & classify_sentiment, 
               as we will call them in testing
    
    Sentiment Labels: 1 (positive); -1 (negative); 0 (neutral)
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import matplotlib.pyplot as plt
from matplotlib.style import use     # Requires matplotlib to create plots.
import numpy as np
# from pyrsistent import T    # Requires numpy to represent the numbers

def draw_plot(r, f, imgname):
    # Data for plotting
    x = np.asarray(r)
    y = np.asarray(f)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Rank (log)', ylabel='Frequency (log)',
        title='Word Frequency v.s. Rank (log)')
    ax.grid()
    fig.savefig(f"../plots/{imgname}")
    plt.show()


import re
# import math
# import collections

import json
import codecs
from tkinter import E

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))

def cal_accu(preds, golds):
    compare = [1 if p == g else 0 for p,g in zip(preds, golds)]
    return sum(compare) / len(compare)


class SentimentClassifier:
    def __init__(self, path="../data/labeled_tweets.json", emoticon_dir="../data/emoticons.txt", lexicon_dir="../data/lexicon_sentiment.txt"):
        ## Load emoticon list
        with open(emoticon_dir, 'r', encoding='utf-8') as f:
            self.emoticons = f.read().strip().split('\n')   # emoticon list
        
        ## Load lexicon sentiment list
        with open(lexicon_dir, 'r', encoding='utf-8') as f:
            data = [d.strip().split('\t') for d in f.read().strip().split('\n')]
        self.lexicon_sentiment = {d[0]: int(d[1]) for d in data}    # lexicon sentiment list
        self.emoticon_lexicon_sentiment = [0, 0, 0, 0.03056768558951965, -1.0, 0.42857142857142855, 1.0, 0.07692307692307693, 0, -0.07692307692307693, -0.5257731958762887, 0, 0, -0.52, -0.3333333333333333, -0.25, 0, 0.0, -0.5, 0.2222222222222222, -0.12903225806451613, 0.07692307692307693, -0.1111111111111111, -0.3333333333333333, 0.2]
        
        ## Load tweet corpus
        raw_tweets = json_load(path)   # raw data (labeled) of tweets
        self.tweet_text = [x["review"] for x in raw_tweets]
        self.raw_tweets = raw_tweets
        
    
    def judge_emoticon(self, word):
        '''Determine whether a word(token) is an emoticon'''
        for emoticon in self.emoticons: 
            if (word == emoticon): return True
        return False

    def clean_text(self, text): 
        username_pattern = re.compile("@\w+")
        email_pattern = re.compile("[\w.]+@[\w]+[.][\w.]+")
        weblink_pattern = re.compile("(?:[\w.\/:]+?:\/\/|[wW]{3}[.])[\w.\/:]+")

        found = []
        found += re.findall(username_pattern, text)
        found += re.findall(email_pattern, text)
        found += re.findall(weblink_pattern, text)

        text = re.sub(username_pattern, "", text)
        text = re.sub(email_pattern, "", text)
        text = re.sub(weblink_pattern, "", text)

        return text
    
    def find_emoticons(self, text): 
        found_emoticons = []
        for emoticon in self.emoticons: 
            result = text.find(emoticon)
            while result != -1: 
                text = text[:result] + text[result+len(emoticon): ]
                found_emoticons += [emoticon]
                result = text.find(emoticon)

        return text, found_emoticons

    
    def process_text(self, text):
        '''Tokenization & Clean & Extract Emoticons on the input text
        Please also clean the tweet data by removing some noise (i.e. email, weblink).
        
        You need to extract the emoticons and distinguish them from other tokens,
        as in the basic implementation of classify_sentiment, only the non-emoticon 
        tokens are utilized.

        [In] original text of a tweet
        [Out] a sample of the review displayed in dict
        E.g.
        [In] text='I like it. :)'
        [Out] return {'raw': 'I like it. :)', 'text': ['I', 'like', 'it', '.'], 'emoticons': [':)']}
        '''
        
        tokens_pattern = re.compile("((?:\d+[\.:\/]\d+)|(?:[a-zA-Z]+'[a-zA-Z]+)|(?:[a-zA-Z]+')|(?:&lt;?3)|(?:[a-zA-Z-]+)|\.|\?|\!|;|:|\/|,|\")")

        cleaned = self.clean_text(text)
        rest, emoticons = self.find_emoticons(cleaned) 
        clean_text = re.findall(tokens_pattern, rest)
        # print("Original:", rest)
        # print("Tokenised:", tokenised)
        
        return {'raw': text, 'text': clean_text, 'emoticons': emoticons}

    def classify_sentiment(self, sample, threshold=0.1, 
                           use_emoticon=False, ratio=0.5):
        '''Utilize lexicon sentiment (and emoticon) for sentiment analysis

        sample: the input tweet (with tokens, emoticons, tokens excluding emoticons)
        threshold: threshold to decide whether to choose 1/-1 or 0
        use_emoticon: whether to use emoticons for sentiment analysis
        ratio: weights of emoticons when making the final decision

        E.g.
        [In] 
        sample={'raw': 'I like it. :)', 'text': ['I', 'like', 'it', '.'], 'emoticons': [':)']}
        threshold=0.1  use_emoticon=True  ratio=0.25
        [Out] return 1
        '''
        ## Get the lexicon sentiment for each non-emoticon token
        word_sentiment = []
        for word in sample['text']:
            if word in self.lexicon_sentiment:
                if self.lexicon_sentiment[word] == 1:
                    word_sentiment.append(1)
                elif self.lexicon_sentiment[word] == -1:
                    word_sentiment.append(-1)
                else:
                    word_sentiment.append(0)
            else:
                word_sentiment.append(0)

        assert len(sample['text']) == len(word_sentiment)
        
        ## Calculate the avg score as polarity
        lx_label = sum(word_sentiment) / len(word_sentiment)

        ## Whether to utilize the information from emoticons
        if use_emoticon:
            emo_sentiment = []
            ###################################################
            ### Utilize emoticon to improve ###################
            for emo in sample['emoticons']:
                emo_index = self.emoticons.index(emo)
                emo_sentiment += [ self.emoticon_lexicon_sentiment[emo_index] ]

            if len(sample["emoticons"]) == 0:  # Set emoticon sentiment to one element of 0. To avoid division by 0. 
                emo_sentiment = [0]
            ###################################################
            
            em_label = sum(emo_sentiment) / len(emo_sentiment)
            label = em_label * ratio + lx_label * (1 - ratio)
        else:
            label = lx_label

        ## Generate the discrete labels using the threshold 
        label = 1 if label > threshold else label
        label = -1 if label < -threshold else label
        label = 0 if label not in [-1, 1] else label

        return label

    def plot_emoticon_frequency(self):
        '''
        Plot relative frequency versus rank of emoticon to check
        Zipf's law
        You may want to use matplotlib and the function shown 
        above to create plots
        Relative frequency f = Number of times the emoticon occurs /
                                Total number of emoticon tokens
        Rank r = Index of the emoticon according to emoticon occurence list
        '''

        all_emoticons = []
        for tweet in self.tweet_text: 
            processed_text = self.process_text(tweet)
            all_emoticons += processed_text["emoticons"]

        emoticon_set = set(all_emoticons)
        emoticon_list = list(emoticon_set)
        freq = [all_emoticons.count(x) for x in emoticon_list]

        # for i in range(0, len(emoticon_list)): 
        #     print(emoticon_list[i], freq[i])

        relative_freq = np.divide(freq, len(all_emoticons))
        log_relative_freq = np.log(relative_freq)

        # print(log_relative_freq)
        log_relative_freq.sort() # sorted but inverse. np cannot sort decreasing for some reason
        
        ranks = range(len(emoticon_list), 0, -1) # ranks arranged but inverse also 
        log_ranks = np.log(ranks)

        draw_plot(log_ranks, log_relative_freq, "3")





# s = SentimentClassifier()
# # s.plot_emoticon_frequency()

# outputs = []
# zero_predictions = 0

# for tweet in s.raw_tweets: 
#     label = tweet["label"]
#     text = tweet["review"]
#     index = tweet["id"]
#     sample = s.process_text(text)
#     pred_label = s.classify_sentiment(sample, threshold=0.15, use_emoticon=True, ratio=0.5)
    
#     sample["id"] = index
#     sample["pred_label"] = pred_label
#     sample["label"] = label

#     sample["correct"] = (pred_label == label)

#     if pred_label == 0: zero_predictions+=1

#     outputs += [sample]

# no_emot_acc = []
# emot_acc = []

# emoticon_dict = dict()
# emoticon_sentiment = [0]*len(s.emoticons) # Empty list of the number of emoticons accepted

# for output in outputs: 
#     if len(output["emoticons"]) == 0: # For cases with no (detected) emoticons
#         no_emot_acc += [output["correct"]]
#     else: 
#         emot_acc += [output["correct"]]

#         for emot in output["emoticons"]: 
#             if emot in emoticon_dict.keys(): emoticon_dict[emot] += [output["label"]] 
#             else: emoticon_dict[emot] = [output["label"]] 

# --- For finding the sentiment of emoticons --- 
# for (e,v) in emoticon_dict.items():
#     sentiment_mean = np.mean(v)
#     index = s.emoticons.index(e)
#     emoticon_sentiment[index] = sentiment_mean
#     print("----", e, v, sentiment_mean)


# all_acc = no_emot_acc + emot_acc

# emoticon_accuracy = emot_acc.count(True)/len(emot_acc)
# no_emoticon_accuracy = no_emot_acc.count(True)/len(no_emot_acc)
# all_acc = all_acc.count(True)/len(all_acc)

# print("Predict zero!", zero_predictions)
# print(len(no_emot_acc), len(emot_acc))

# print(emoticon_accuracy, no_emoticon_accuracy, all_acc)


if __name__ == '__main__':
    datadir = '../data/labeled_tweets.json'
    emoticon_dir = '../data/emoticons.txt'
    lexicon_dir = '../data/lexicon_sentiment.txt'

    labeled_reviews = json_load(datadir)
    reviews = [d['review'] for d in labeled_reviews]
    GT_labels = [d['label'] for d in labeled_reviews]

    classifier = SentimentClassifier(datadir, emoticon_dir, lexicon_dir)

    samples = [classifier.process_text(review) for review in reviews]
    labels_wo_emoticons = [classifier.classify_sentiment(sample) for sample in samples]
    labels_wi_emoticons = [classifier.classify_sentiment(sample, use_emoticon=True) for sample in samples]

    print('Accu without emoticons: {:.2f}%'.format(100 * cal_accu(labels_wo_emoticons, GT_labels)))
    print('Accu with emoticons: {:.2f}%'.format(100 * cal_accu(labels_wi_emoticons, GT_labels)))