'''
    NUS CS4248 Assignment 1 - Objective 2 (Regular Expression, Edit Distance)

    Class EditDistanceCalculator & RegexChecker for handling Objective 2

    Important: please strictly comply with the input/output formats for
               the methods of calculate_edit_distance & approximate_matches, 
               as we will call them in testing
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import re
# from tracemalloc import start
import numpy as np


class EditDistanceCalculator:
    def __init__(self):
        # TODO Modify the code here
        pass
    
    def calculate_edit_distance(self, source, target):
        '''Calculate the edit distance from source to target

        E.g.
        [In] source="ab" target="bc"
        [Out] return 2
        '''
        
        distance_table = np.empty((len(source)+1, len(target)+1)) # Initialise table with space for each character plus the empty string
        distance_table[0,:] = range(0, len(target)+1)
        distance_table[:,0] = range(0, len(source)+1)

        for i in range(1, len(source)+1):
            for j in range(1, len(target)+1): 
                add = distance_table[i-1,j] + 1
                delete = distance_table[i,j-1] + 1
                
                same = (source[i-1] == target[j-1]) # Whether the current letters being considered are the same
                if same: substitute = distance_table[i-1,j-1]
                else: substitute = distance_table[i-1,j-1] + 2

                distance_table[i,j] = min(add, delete, substitute)

        return distance_table[len(source), len(target)] 

class RegexChecker:
    # def __init__(self, regex=r"\b(?![^ ]*([^ ])([^ ])\2\1[^ ]*)([^ \n])[^ \n]*\3\b"):
    def __init__(self, regex):

        '''The value of regex here should be fixed as the R_3 you've solved'''
        self.regex = regex
        self.abba_regex = r"(([^ ])([^ ])\3\2)"
        self.calc = EditDistanceCalculator()
    
    def matches(self, word):
        '''Return whether a word is (exactly) matched by the regex'''
        match = re.fullmatch(self.regex, word)

        if match: return True
        else: return False

    def matches_abba(self, word):
        '''Return if the word contains an 'abba' pattern'''
        match = re.search(self.abba_regex, word)

        if match: return match.span()
        else: return None

    # def get_edit_distance(self, test, original): 
    #     '''Use the EditDistanceCalculator to obtain the edit distance between
    #     two diffferent words. 
    #     '''
    #     dist = self.calc.calculate_edit_distance(test, original)
    #     return dist

    def approximate_matches(self, word, k=2):
        '''Return whether a word can be matched by the regex within k 
        errors (edit distance)
        You can assume that the word is got from a corpus which has 
        already been tokenized.

        E.g.
        [In] word="blabla"
        [Out] return True
        '''
        
        if self.matches(word): return True # Good!
        if k==0: return False # No edit distance to make a match 

        # k > 0 and it does not match. 

        # Identify abba matches
        matches = self.matches_abba(word)
        
        # If no abba patterns:
        if matches == None:  
        
            # Deletion can never create new abba patterns
            if len(word) > 2: 
                new_word = word[0:-1] # Delete 1 from the back, k is at least 1
                if self.matches(new_word): return True
                new_word = word[1:] # Delete 1 from the front, k is at least 1
                if self.matches(new_word): return True 

            # If we cannot get there by deleting, try to add 
            new_word = word + word[0] # Add 1 to the back, k is at least 1
            if self.matches(new_word): return True
            new_word = word[-1] + word # Add 1 to the front, k is at least 1
            if self.matches(new_word): return True

            # In the odd case that adding chars on the front/back creates abbas. e.g. "bbabaa"
            # Recurse downwards on a shorter string
            result1 = self.approximate_matches(word[1:], k-1) 
            result2 = self.approximate_matches(word[:-1], k-1)
            if result1 or result2 : return True
            
        else:  # try to remove abba matches

            start_index = matches[0] # Starting index of the abba pattern
            end_index = matches[1] # Ending index of the abba pattern

            # If abba occurs at the start or end of the sequence
            if start_index == 0: 
                if self.matches(word[1:]): return True
                result = self.approximate_matches(word[1:], k-1) # Break the abba by removing the first char
                if result: return True
            if end_index == len(word): 
                if self.matches(word[:-1]): return True
                result = self.approximate_matches(word[:-1], k-1) # Break the abba by removing the last char
                if result: return True
            
            abba = word[start_index:start_index+4]

            # To avoid repeating chars which might create more patterns 
            used_chars = set()
            used_chars.add(abba[0])
            used_chars.add(abba[1])

            if (start_index>0): used_chars.add(word[start_index-1]) # add the char 1 before abba
            if (start_index+4<len(word)): used_chars.add(word[start_index+4]) # add the char 1 after abba
            if (start_index+5<len(word)): used_chars.add(word[start_index+5]) # add the char 2 aftter abba
            if (start_index+6<len(word)): used_chars.add(word[start_index+6]) # add the char 3 aftter abba


            # Possible chars left to sub
            # Check 3 chars after the abba - abb_aac 
            possible_chars = set(['a','b','c','d','e','f','g']) # Max 4 will repeat, we have 5 available. 
            possible_chars = possible_chars.difference(used_chars) # Get avail chars to work with 
            add_char = possible_chars.pop() # Randomly get a usable char

            if abba[0] == abba[1]: # The pattern is actually aaaa. 
                # Break up with aaa_a. Helps deal with instances like "aaaaa"
                # since regex matches from the front
                new_word = word[0:start_index+3] + add_char + word[start_index+3:] 
                if self.matches(new_word): return True
            else: # It is abba
                # Break up by truncating to aba
                new_word = word[0:start_index+2] + word[start_index+3:] 
                if self.matches(new_word): return True
            
            # Recurse after breaking up the abba pattern
            result = self.approximate_matches(new_word, k-1) 
            if result: return True

        return False # If everything cannot then return false

        

if __name__ == '__main__':
    ## test the edit-distance-calculator ##
    calculator = EditDistanceCalculator()

    test_cases = [['ab', 'bc']]
    for case in test_cases:
        source, target = case
        distance = calculator.calculate_edit_distance(source, target)
        
        print(distance) # 2.0
    
    ## test the regex-checker ##
    R3 = r"(?:(?<=\s)|(?<=^))(?!\S*(\S)(\S)\2\1\S*)(\S)\S*\3(?:(?<=\s)|(?<=$))"    # set the R3 here
    checker = RegexChecker(regex=R3)

    test_cases = ['blabla', 'aaaaaaaaaaaa', 'aaaabaaa', '-++-']
    for case in test_cases:
        flag = checker.approximate_matches(case, k=2)
        
        print(flag) # True
# r = RegexChecker()
# print(r.approximate_matches("aaaabaaa", 0))
