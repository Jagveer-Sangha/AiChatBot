import torch
import torch.nn as nn

# Used for neural network and deep learning
# Feed Forward Neural Net


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Created 3 different linear layers of the neural net
        # first arg is input second is output
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, num_classes)
        # Activation function for inbetween
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax
        return out
        
