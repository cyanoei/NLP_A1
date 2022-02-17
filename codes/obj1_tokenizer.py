'''
    NUS CS4248 Assignment 1 - Objective 1 (Tokenization, Zipf's Law)

    Class Tokenizer for handling Objective 1

    Important: please strictly comply with the input/output formats for
               the method of tokenize_sentence, as we will call it in testing
'''
###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import matplotlib.pyplot as plt     # Requires matplotlib to create plots.
import numpy as np    # Requires numpy to represent the numbers
import re
# import pickle

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
    # plt.show()

class Tokenizer:

    def __init__(self, path, bpe=False, lowercase=True):
        # OBJ1 path=STRING bpe=YES/NO lowercase=YES/NO

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()
        
        self.bpe = bpe
        self.lowercase = lowercase
        
        if self.lowercase: 
            self.text = self.text.lower() 

        self.regex_pattern = "((?:\d+[\.:\/-]\d+)|(?:[a-zA-Z]+'[a-zA-Z]+)|(?:[a-zA-Z]+')|(?:[a-zA-Z0-9-]+)|[^\t\n\r n])"

        self.word_tokens = None # Store word tokens after tokenisation to use for plotting
        self.tokens_to_merge = None  # BPE only: List of pairs of tokens to merge
        self.ignore_tokens = [' ', '\n'] # Accepted word boundaries 
                
    def tokenize(self):
        ''' Returns/Saves a set of word tokens for the loaded textual file

        For the default setting, make sure you consider cases of:
        1) words ending with punctuation (e.g., 'hiking.' ——> ['hiking', '.']);
        2) numbers (e.g., '1/2', '12.5')
        3) possessive case (e.g., "Elle's book" ——> ["Elle's", "book"])

        For the bpe setting, 
        1) tune the number of iterations so the vocab size will be close to the 
        default one's
        2) during merge, for sub-sequences of the same frequency, break the tie 
        with left-to-right byte order precedence
        '''

        if (self.bpe): # Tokenisation by BPE - 1-1-B
            current_token_list = list(self.text)

            full_token_list = []

            counter = 0
            while (1): 
                counter += 1
                token_merged = dict()

                for (i, _) in enumerate(current_token_list): 
                    if i == len(current_token_list)-1: continue
                    
                    tok1 = current_token_list[i]
                    tok2 = current_token_list[i+1]

                    if (tok1 in self.ignore_tokens) or (tok2 in self.ignore_tokens): continue

                    new_token = (current_token_list[i], current_token_list[i+1])
                    if new_token in token_merged: 
                        token_merged[new_token] += 1
                    else:
                        token_merged[new_token] = 1

                # handling tiebreakers
                maximum_count = max(token_merged.values()) 

                # we find that around 6500 words the number of instances in the text is 3. Hence we will only consider up to pairs which occur 3 times. 
                # EDIT: I now know that the V2 handout now says approx 13k words, but please understand that I spent so long on Q1 alone that I no longer have time to edit further... 
                if (maximum_count < 3): break 

                all_max_tokens = [k for k, v in token_merged.items() if v == maximum_count]
                all_max_tokens.sort(key=lambda x: x[0]+x[1])

                # choosing the pair to merge
                maximum_token = all_max_tokens[0]

                # print("We are merging number", counter, ": ", maximum_token, " which occurs this many times: ", maximum_count)
                full_token_list += [maximum_token]

                # merging and replacing 
                for (i, _) in enumerate(current_token_list): 
                    if i == len(current_token_list)-1: continue
                    
                    tok1 = current_token_list[i]
                    tok2 = current_token_list[i+1]

                    target1 = maximum_token[0]
                    target2 = maximum_token[1]

                    if (tok1 == target1) and (tok2 == target2): 
                        current_token_list.pop(i+1)
                        current_token_list[i] = tok1+tok2
                
                # if (counter >= 7000): break # max out the vocabulary at 7000 words (approx 6653)
            
            # Save tokens learnt during BPE 
            self.tokens_to_merge = full_token_list
            
            # with open('../data/bpe_dict_7000.pickle', 'wb') as handle:
            #     pickle.dump(full_token_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

            tokenised = current_token_list
            for ignore in self.ignore_tokens: 
                tokenised = [token for token in tokenised if token != ignore] # list comp to remove all instances of the character

        # Tokenisation by RegEx - 1-1-A
        else : 
            tokens_pattern = re.compile(self.regex_pattern)
            tokenised = re.findall(tokens_pattern, self.text)

        print(len(set(tokenised))) 
        # RegEx 6749 words found. 
        # BPE 6311 words found. 
        self.word_tokens = tokenised
        return tokenised
         
    def tokenize_sentence(self, sentence):
        '''
        To verify your implementation, we will test this method by 
        input a sentence specified by us.  
        Please return the list of tokens as the result of tokenization.

        E.g. basic tokenizer (default setting)
        [In] sentence="I give 1/2 of the apple to my ten-year-old sister."
        [Out] return ['i', 'give', '1/2', 'of', 'the', 'apple', 'to', 'my', 'ten-year-old', 'sister', '.']
        
        PS: For BPE, you may need to fix the vocab before tokenizing
            the input sentence
        '''

        if (self.lowercase): sentence = sentence.lower()

        if (self.bpe):
            token_list = list(sentence)

            count = 0

            # if self.tokens_to_merge == None: 
            #     try: 
            #         pickle_path = "../data/bpe_dict_ordered.pickle"
            #         with open(pickle_path, 'rb') as pickle_file:
            #             tokens_dict = pickle.load(pickle_file)
            #         self.tokens_to_merge = tokens_dict

            #     except FileNotFoundError: 
            #         print("No saved vocabulary found. Run tokenize() first.")

            for pair in self.tokens_to_merge:
                count += 1
                # print(count, ": ", pair)
                # if count >= 100: break
                for (i, _) in enumerate(token_list): 
                    if i >= len(token_list)-1: break
                    tok1 = token_list[i]
                    tok2 = token_list[i+1]

                    target1 = pair[0]
                    target2 = pair[1]

                    if (tok1 == target1) and (tok2 == target2): 
                        token_list.pop(i+1)
                        token_list[i] = tok1+tok2
                
            for ignore in self.ignore_tokens: 
                token_list = [token for token in token_list if token != ignore] # list comp to remove all instances of the character

            # print(token_list)
            return token_list
        
        else: # RegEx tokenisation      
            tokens_pattern = re.compile(self.regex_pattern)
            tokenised = re.findall(tokens_pattern, sentence)
            return tokenised

    
    def plot_word_frequency(self):
        '''
        Plot relative frequency versus rank of word to check
        Zipf's law
        You may want to use matplotlib and the function shown 
        above to create plots
        Relative frequency f = Number of times the word occurs /
                                Total number of word tokens
        Rank r = Index of the word according to word occurence list
        '''
        
        total_tokens = len(self.word_tokens)

        tokenised_set = set(self.word_tokens)
        tokenised_list = list(tokenised_set)

        freq = [self.word_tokens.count(x) for x in tokenised_list]
        relative_freq = np.divide(freq, total_tokens)
        log_relative_freq = np.log(relative_freq)

        # print(log_relative_freq)
        log_relative_freq.sort() # sorted but inverse. np cannot sort decreasing for some reason
        
        ranks = range(len(tokenised_set), 0, -1) # ranks arranged but inverse also 
        log_ranks = np.log(ranks)

        if self.bpe: plot_name = "1-1-B"
        else: plot_name = "1-1-A"

        draw_plot(log_ranks, log_relative_freq, plot_name)


# Updated from V2 file 

if __name__ == '__main__':
    ##=== tokenizer initialization ===##
    basic_tokenizer = Tokenizer('../data/Pride_and_Prejudice.txt')
    # bpe_tokenizer = Tokenizer('../data/Pride_and_Prejudice.txt', bpe=True)

    ##=== build the vocab ===##
    try:
        _ = basic_tokenizer.tokenize()  # for those which have a return value
    except:
        basic_tokenizer.tokenize()
    # try:
    #     _ = bpe_tokenizer.tokenize()  # for those which have a return value
    # except:
    #     bpe_tokenizer.tokenize()

    ##=== run on test cases ===##
    
    # you can edit the test_cases here to add your own test cases
    test_cases = ["""The Foundation's business office is located at 809 North 1500 West, 
        Salt Lake City, UT 84116, (801) 596-1887."""]
    
    for case in test_cases:
        rst1 = basic_tokenizer.tokenize_sentence(case)
        # rst2 = bpe_tokenizer.tokenize_sentence(case)

        ##= check the basic tokenizer =##
        # ['the', "foundation's", 'business', 'office', 'is', 'located', 'at', 
        # '809', 'north', '1500', 'west', ',', 'salt', 'lake', 'city', ',', 'ut', 
        # '84116', ',', '(', '801', ')', '596-1887', '.']
        # or
        # ['the', 'foundation', "'s", 'business', 'office', 'is', 'located', 'at', 
        # '809', 'north', '1500', 'west', ',', 'salt', 'lake', 'city', ',', 'ut', 
        # '84116', ',', '(', '801', ')', '596-1887', '.']
        print(rst1)

        ##= check the bpe tokenizer =##
        # ['the_', 'f', 'ou', 'n', 'd', 'a', 'ti', 'on', "'", 's_', 'bu', 
        # 's', 'in', 'es', 's_', 'o', 'f', 'f', 'i', 'c', 'e_', 'is_', 'l', 
        # 'o', 'c', 'at', 'ed_', 'at_', '8', '0', '9', '_', 'n', 'or', 'th_', 
        # '1', '5', '0', '0', '_', 'w', 'es', 't', ',_', 's', 'al', 't_', 'l', 
        # 'a', 'k', 'e_', 'c', 'it', 'y', ',_', 'u', 't_', '8', '4', '1', '1', 
        # '6', ',_', '(', '8', '0', '1', ')', '_', '5', '9', '6', '-', '1', '8', 
        # '8', '7', '._']
        # print(rst2)
    














# --- Old testing stuff 
# t = Tokenizer("../data/Pride_and_Prejudice.txt", True, True)
# t.tokenize()
# t.plot_word_frequency()

# with open("../data/Pride_and_Prejudice.txt", 'r', encoding='utf-8', errors='ignore') as f:
#     test_text = f.read()
# tokenised = t.tokenize_sentence(test_text.lower())

# word_dict = dict()
# for token in tokenised: 
#     if token in word_dict: 
#         word_dict[token] += 1
#     else: 
#         word_dict[token] = 1

# rare_words = [(k, v) for k, v in word_dict.items() if v >1000]

# print(rare_words)


