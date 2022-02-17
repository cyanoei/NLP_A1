from nltk.tokenize import sent_tokenize


with open('../data/Pride_and_Prejudice.txt', 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            # raw_corpus = raw_corpus.lower() # Consider whether we want to lowercase the text    
            text = raw_text.replace("\n", "")

# text = "Hello. Mrs. Lee and rr. Lee (8800-12) New York 1.2.3. 23? mr. m. a-a-_ a? rev. And then; alas. ??!!?!?!?! \" \'fwoijweio\' \"" 
output = sent_tokenize(text, language='english')

i=0
for o in output: 
    i+=1
    print(o)
    if (i >20): break