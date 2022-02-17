import pickle

with open("../data/bpe_dict.pickle", 'rb') as pickle_file: 
    bpe_pairs = pickle.load(pickle_file)

path = "../data/Twinkle_Toes.txt"
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

text = text.lower()

token_list = list(text)
print(token_list[100])

count = 0
for pair in bpe_pairs:
    count += 1
    # print(count, ": ", pair)
    # if count >= 100: break
    for (i, c) in enumerate(token_list): 
        if i >= len(token_list)-1: break
        tok1 = token_list[i]
        tok2 = token_list[i+1]

        target1 = pair[0]
        target2 = pair[1]

        if (tok1 == target1) and (tok2 == target2): 
            token_list.pop(i+1)
            token_list[i] = tok1+tok2
    
    if (count%500 == 0): print(token_list[0:200])

print(token_list)
