import json

from torch._C import device
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

"""PyTorch imports"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Import from model.py
from model import NeuralNet

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
# print(tags)

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


# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Testing
print(input_size, len(all_words))
print(output_size, tags)

dataset = ChatDataset()
# Multi threading. Makes the loading quicker
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Checks for gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = NeuralNet(input_size, hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        # will get the predicted output and actual labels
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        # calculates the back propogation
        loss.backward()
        optimizer.step()

    if(epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

# Overall final loss is lower if data is flattened
print(f'final loss, loss={loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
