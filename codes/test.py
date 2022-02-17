from numpy import full
import pickle

path = "../data/Pride_and_Prejudice.txt"
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

text = text.lower()
current_token_list = list(text)
ignore_tokens = [' ', '\n']

full_token_list = []

counter = 0

while (1): 
    counter += 1
    token_merged = dict()

    for (i, c) in enumerate(current_token_list): 
        if i == len(current_token_list)-1: continue
        
        tok1 = current_token_list[i]
        tok2 = current_token_list[i+1]

        if (tok1 in ignore_tokens) or (tok2 in ignore_tokens): continue

        new_token = (current_token_list[i], current_token_list[i+1])
        if new_token in token_merged: 
            token_merged[new_token] += 1
        else:
            token_merged[new_token] = 1

    
    if len(token_merged) < 1: break
    # maximum_token = max(token_merged, key=token_merged.get)
    maximum_count = max(token_merged.values()) # handling tiebreakers
    all_max_tokens = [k for k, v in token_merged.items() if v == maximum_count]
    all_max_tokens.sort(key=lambda x: x[0]+x[1])

    maximum_token = all_max_tokens[0]

    print("We are merging number", counter, ": ", maximum_token, " which occurs this many times: ", maximum_count)
    full_token_list += [maximum_token]

    for (i, c) in enumerate(current_token_list): 
        if i == len(current_token_list)-1: continue
        
        tok1 = current_token_list[i]
        tok2 = current_token_list[i+1]

        target1 = maximum_token[0]
        target2 = maximum_token[1]

        if (tok1 == target1) and (tok2 == target2): 
            current_token_list.pop(i+1)
            current_token_list[i] = tok1+tok2
    
    # Store data (serialize)
    with open('../data/bpe_dict.pickle', 'wb') as handle:
        pickle.dump(full_token_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Load data (deserialize)
    # with open('../data/bpe_dict.pickle', 'rb') as pickle_file:
    #     tokens_dict = pickle.load(pickle_file)