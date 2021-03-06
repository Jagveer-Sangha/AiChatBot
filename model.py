import torch
import torch.nn as nn

# Used for neural network and deep learning
# Feed Forward Neural Net


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # Created 3 different linear layers of the neural net
        # first arg is input second is output
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Activation function for inbetween
        self.relu = nn.ReLU()

        # Flattens multidimensional data into 1D
        # self.f1 = nn.Flatten()

    def forward(self, x):
        # out = self.f1(x)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax
        return out
