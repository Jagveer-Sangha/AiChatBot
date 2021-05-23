import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

"""PyTorch imports"""
import torch
from torch.utils.data import Dataset, DateLoader, dataset
import torch.nn as nn


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

x_train = []
y_train = []
for(pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

# Creating numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples

    # Hyperparameters
    batch_size = 8

    dataset = ChatDataset()
    # Multi threading. Makes the loading quicker
    train_loader = DateLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_works=2)
