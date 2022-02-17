'''
    NUS CS4248 Assignment 1 - Objective 4 (n-gram Language Model)

    Class NgramLM for handling Objective 4

    Important: please strictly comply with the input/output formats for
               the methods of generate_word & generate_text & perplexity, 
               as we will call them in testing
    
    Sentences for Task 4-B:
    1) "They had now entered a beautiful walk by"
    2) "The snakes entered a beautiful walk by the buildings."
    3) "They had now entered a walk by"
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

from lib2to3.pgen2 import token
import random, math
# from lib2to3.pgen2 import token
import re
# from nltk.util import ngrams
import numpy as np

class NgramLM(object):

    def __init__(self, path, n, k=1.0):
        '''
            Initialize your n-gram LM class

            Parameters:
                n (int) : order of the n-gram model
                k (float) : smoothing hyperparameter

        '''
        # Initialise other variables as necessary
        # TODO Write your code here
        self.n = n
        self.k = k

        self.word_count_dict = {}
        self.bigram_count_dict = {}

        self.token_pattern = re.compile(r"((?:\d+[\.:\/-]\d+)|(?:[a-zA-Z]+'[a-zA-Z]+)|(?:[a-zA-Z]+')|(?:[a-zA-Z0-9-]+)|[^\t\n\r n])")
        self.sentence_pattern = re.compile(r"((?!Mr.|Mrs.|Miss.|Ms.|Dr.|[A-Z]\.|\d+\.\d+)(?:\b[\w]+)[\'\"\”\)]*[\.\?\!]+[\'\"\”\)]*)")

        self.vocab = None
        self.total_vocab_count = 0 
        self.ngrams = None
        self.ngram_counts = None


        # Read file and process the vocab/ngrams
        self.read_file(path)

        
    def split_sentences(self, text): 
        sentence_list = []
        sentence_end = re.search(self.sentence_pattern, text)

        while (sentence_end != None):
            end_index = sentence_end.span()[1]
            sentence_list += [text[:end_index]]
            text = text[end_index:] 
            sentence_end = re.search(self.sentence_pattern, text)
        
        return sentence_list


    def update_corpus(self, text):
        ''' Updates the n-grams corpus based on text 
        
        PS: This method is a suggested method to implement 
            which you may call in the method of read_file 
        '''
        self.get_ngrams(text) # Updates self.ngrams and self.vocab directly
        # self.get_vocabulary(text) 
        
        #  --- Get the full bigram counts if n=2
        if self.n == 1 : return # No need to continue if only 1
        ngram_set = set(self.ngrams) # Get set of all bigrams

        for ng in self.ngrams: 
            if (ng in self.bigram_count_dict.keys()): self.bigram_count_dict[ng] += 1
            else: self.bigram_count_dict[ng] = 1
            # self.bigram_count_dict[ng] = self.ngrams.count(ng)
        
        # print(self.bigram_count_dict)

    def read_file(self, path):
        ''' Read the file and update the corpus  
        
        PS: This method is a suggested method to implement 
            which you may call in the method of __init__ 
        '''
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            # raw_corpus = raw_corpus.lower() # Consider whether we want to lowercase the text    
            raw_text = raw_text.replace("\n", "")

            self.update_corpus(raw_text)

    def get_ngrams(self, text):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] 
        
            the sequence context represent the ngram and the word is its last word
        
        Hint: to get the ngrams, you may need to first get splited sentences from the corpus.

        PS: This method is a suggested method to implement 
            which you may call in the method of update_corpus 
        '''
        sentences = self.split_sentences(text)
        ngrams = []
        all_tokens = []
        for s in sentences: 
            tokenised = re.findall(self.token_pattern, s)
            tokenised = self.add_padding(tokenised)
            
            all_tokens += tokenised # Collect all text with padding 
            
            if self.n == 1: 
                # Get just individual words
                ngrams += [ (x,) for x in tokenised] 
            else: 
                # Get pairs of words
                ngrams += [ (tokenised[i], tokenised[i+1]) for i, _ in enumerate(tokenised) if i < len(tokenised)-1]

        self.ngrams = ngrams
        self.ngram_counts = len(ngrams)
        # print("Ngram counts: ", self.ngram_counts) # 149705 (bigrams)


        # Get vocab while we're at it 
        self.vocab = set(all_tokens) 
        self.total_vocab_count = len(self.vocab)
        # print("Total vocab: ", self.total_vocab_count) # 7395 words

        # Populate the word count dictionary. '~' is included. 
        for token in all_tokens: 
            if (token in self.word_count_dict.keys()): self.word_count_dict[token] += 1
            else: self.word_count_dict[token] = 1

    def add_padding(self, tokens): 
        '''  Returns padded text - for individual sentences 
        The goal of the method is to pad start token(s) 
        for each sentence so that we can get ngram like 
        a 2-gram '~ I' for a sentence 'I like NUS.'

        PS: This method is a suggested method to implement 
            which you may call in the method of ngrams 
        
        '''
        # Use '~' as your padding symbol
        # It only says start token so... I will use only start token 
        return ["~"] + tokens 
        

    # def get_vocabulary(self, text):
    #     ''' Returns the vocabulary as set of words 
        
    #     PS: This method is a suggested method to implement 
    #         which you may call in the method of update_corpus 
    #     '''
    #     tokenised = re.findall(self.token_pattern, text) # Tok the whole text

    #     self.vocab = set(tokenised) 
    #     self.total_vocab_count = len(self.vocab)
    #     print("Total vocab: ", self.total_vocab_count)

    #     # Populate the word count dictionary. 
    #     for v in self.vocab: 
    #         vocab_count = tokenised.count(v)
    #         self.word_count_dict[v] = vocab_count

    def get_next_word_probability(self, text, word):
        ''' Returns the probability of word appearing after specified text 
        
        PS: This method is a suggested method to implement 
            which you may call in the method of generate_word
        '''

        # We look for P(word|rest) which is P(full_text)/P(rest)
        # So only need to get P(word|previous_word for bigrams
        # and P(word) for 1-grams

        total_vocab_count = self.total_vocab_count # Number of all words
        total_vocab_size = len(self.vocab) # Number of all unique words
        smoothed_vocab_count = total_vocab_count + ((total_vocab_size+1) * self.k)
        # print(total_vocab_count, smoothed_vocab_count) ~7k and ~14k

        use_smooth = True
        if (use_smooth): 

            if self.n == 1:

                if word in self.vocab: 
                    prob = (self.word_count_dict[word] + self.k)/smoothed_vocab_count # Smoothed word prob
                else: 
                    prob = (self.k)/ smoothed_vocab_count # Use k as the count of the unknown word
                return prob

            else: # self.n == 2 
                tokenised = re.findall(self.token_pattern, text)

                if (len(tokenised) == 0): last_word = "" # Will not be able to be found in the vocab
                else: last_word = tokenised[-1]

                if last_word in self.vocab: # Last word is known ('~' is part of known vocab)
                    pair = (last_word, word)
                    
                    if (last_word, word) in self.bigram_count_dict.keys():  # If this pair is known
                        pair_count = self.bigram_count_dict[pair] + self.k
                        
                    else:                                                   # If the pair is not known - either unknown pairing or second word unknown 
                        # print("Not present in bigram dict.")
                        pair_count = self.k
                    
                    prob = pair_count/(self.word_count_dict[last_word] + (len(self.vocab)+1)*self.k) # Smoothing: add k more instances of (word, n_in_vocab)
                    if (prob>1): print(pair, pair_count, "Something wrong")
                
                else: # last_word is unknown. Revert to 1-grams. 
                    if word in self.vocab: 
                        # Just find P(word) in the corpus
                        prob = (self.word_count_dict[word] + self.k)/smoothed_vocab_count # Smoothed word prob
                    else: 
                        prob = (self.k)/ smoothed_vocab_count # Use k as the count of the unknown word
                return prob

        else: # not smooth    
            if self.n == 1:

                if word in self.vocab: 
                    prob = (self.word_count_dict[word] + self.k)/total_vocab_count # Smoothed word prob
                else: 
                    prob = (self.k)/ total_vocab_count # Use k as the count of the unknown word
                return prob

            else: # self.n == 2 
                tokenised = re.findall(self.token_pattern, text)

                if (len(tokenised) == 0): last_word = "" # Will not be able to be found in the vocab
                else: last_word = tokenised[-1]

                if last_word in self.vocab: # Last word is known ('~' is part of known vocab)
                    pair = (last_word, word)
                    
                    if (last_word, word) in self.bigram_count_dict.keys():  # If this pair is known
                        pair_count = self.bigram_count_dict[pair] 
                        
                    else:                                                   # If the pair is not known - either unknown pairing or second word unknown 
                        # print("Not present in bigram dict.")
                        pair_count = self.k
                    
                    prob = pair_count/self.word_count_dict[last_word]  
                    if (prob>1): print(pair, pair_count, "Something wrong")
                
                else: # last_word is unknown. Revert to 1-grams. 
                    if word in self.vocab: 
                        # Just find P(word) in the corpus
                        prob = (self.word_count_dict[word] + self.k)/total_vocab_count # Smoothed word prob
                    else: 
                        prob = (self.k)/ total_vocab_count # Use k as the count of the unknown word
                return prob


            # else: # Neither word is known to the vocab
            #     total_count = self.total_vocab_count + (len(self.vocab)+1)*self.k
            #     word_prob = self.k
            #     prob = word_prob/total_count
            

        # all_previous = [pairs for pairs in self.bigram_count_dict.keys() if pairs[0] == last_word] # All bigrams with last_word in front 
        # print(all_previous) 
        # all_previous_counts = [self.bigram_count_dict[x] for x in all_previous] # List of counts 

        # total_count = sum(all_previous_counts) + (len(self.vocab)+1) * self.k # Add k to all, whether in corpus or not. Includes <UNK> token. 
                

    def generate_word(self, text):
        '''
        Returns a random word based on the specified text and n-grams learned
        by the model
        PS: This method is mandatory to implement with the method signature as-is.
            We only test one sentence at a time, so you may not need to split 
            the text into sentences here.
        
        [In] string (a sentence or half of a sentence)
        [Out] string (a word)
        '''

        words = []
        word_probs = []

        for word in self.vocab:
            if (word != '~'): # Never generate the start token 
                prob = self.get_next_word_probability(text, word)
                words += [word]
                word_probs += [prob]
        
        word_probs = np.power(word_probs, 2) # Scale up likelihood of choosing a relevant word
        [chosen] = random.choices(words, word_probs, k=1)

        return chosen


    def generate_text(self, length):
        ''' Returns text of the specified length based on the learned model 
        [In] int (length: number of tokens)
        [Out] string (text)


        PS: This method is mandatory to implement with the method signature as-is. 
            The length here is a reasonable int number, (e.g., 3~20)
        '''
        text = "~" # We start with start token
        for i in range(0, length):
            word = self.generate_word(text)
            if re.match(r"[\W]+", word): 
                text += word
            else: text += " " + word 
        
        return text[1:].strip(" ")


    def perplexity(self, text):
        '''
        Returns the perplexity of text based on learned model
        
        [In] string (a short text)
        [Out] float (perplexity) 

        PS: This method is mandatory to implement with the method signature as-is. 
            The output is the perplexity, not the log form you use to avoid 
            numerical underflow in calculation.

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
              Also handle the case when the LM assigns zero probabilities.
        '''
        tokenised = re.findall(self.token_pattern, text)
        tokenised = ['~'] + tokenised # Start with start token
        text_len = len(tokenised)

        perp = 0 # Starting sum for log perplexity (log1=0)

        if (self.n==2): 
            for i in range(0, text_len-1): 
                first_word = tokenised[i]
                second_word = tokenised[i+1]

                raw_prob = self.get_next_word_probability(first_word, second_word)
                log_prob = np.log(raw_prob)
                perp += log_prob

        else: # n == 1
            for i in range(0, text_len): 
                first_word = "" # Blank word, function will handle it fine
                second_word = tokenised[i]

                raw_prob = self.get_next_word_probability(first_word, second_word)
                log_prob = np.log(raw_prob)
                perp += log_prob
        
        perp *= (-1/text_len)
        perp = np.exp(perp)
        return perp

        
        

if __name__ == '__main__':
    LM = NgramLM('../data/Pride_and_Prejudice.txt', n=2, k=1)
    # LM = NgramLM("../data/short_text.txt", n=1, k=1)


    test_cases = [  "They had now entered a beautiful walk by",
                    "The snakes entered a beautiful walk by the buildings.",
                    "They had now entered a walk by", 
                    "The snakes", 
                    "They They They They They", 
                    "We saw a calico cat."    ]

    for case in test_cases:
        word = LM.generate_word(case)
        ppl = LM.perplexity(case)
        print(f'input text: {case}\nnext word: {word}\nppl: {ppl}')
    
    _len = [5, 10, 15]
    for l in _len: 
        text = LM.generate_text(length=l)
        print(f'\npredicted text of length {l}: {text}')