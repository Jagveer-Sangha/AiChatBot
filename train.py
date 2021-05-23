import json
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)

        # We don't want an array of arrays, so extend
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Removes duplicate elements and sorts
all_words = sorted(set(all_words))

# Sorts all separate tags
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []
