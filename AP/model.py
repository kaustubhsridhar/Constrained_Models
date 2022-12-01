import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
import sys
import os

torch.manual_seed(0)

class simpleNet(nn.Module):
    def __init__(self, no_of_inputs = 15, no_of_outputs = 1):
        self.no_of_layers = 1
        self.no_of_hidden_units = 100
        super(simpleNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())

        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        # self.lin_trans.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y


def fit_model(train_data, destination_dir = "./networks/"):

    episode_count = 100
    batch_size = 10

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    filename = os.path.join(destination_dir, "diabetes_model.nt")
    print("Length = ", len(train_data))
    print("Input length = ", len(train_data[0]))
    model = simpleNet(len(train_data[0][0]), 1)
    learning_rate = 1e-3
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)


    for episode in range(episode_count):
        number_of_batches = math.ceil(float(len(train_data)) / float(batch_size))

        total_loss = 0.0

        for batch_index in range(number_of_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(train_data))
            loss_list = []
            optimizer.zero_grad()

            for entry in range(start, end):
                input = np.array(train_data[entry][0][:], dtype = np.float32)
                input = torch.from_numpy(input).to(device)

                target = np.array([train_data[entry][1]], dtype = np.float32)
                target = torch.from_numpy(target).to(device)
                prediction = model.forward(input)

                loss = criterion(prediction, target).to(device)
                loss_list.append(loss)

                # Computing loss for tracking
                total_loss += loss.cpu().detach().numpy()

            loss_combined = torch.stack(loss_list, dim = 0).sum()
            loss_combined.backward()
            optimizer.step()


        print("At episode - ", episode, " avg loss computed - ", total_loss / float(len(train_data)))
        # print("At episode - ", episode, " avg loss computed - ", total_loss )

        torch.save(model, filename)


def test_model(test_data, model = None):

    if not model :
        filename = os.path.join("./networks/", "diabetes_model.nt")
        model = torch.load(filename)


    total_loss = 0.0
    for index in range(len(test_data)):

        input = np.array(test_data[index][0][:], dtype = np.float32)
        input = torch.from_numpy(input)

        target = test_data[index][1]

        prediction = model.forward(input)
        prediction = prediction.cpu().detach().numpy()[0]

        loss = abs( prediction - target)
        total_loss += loss


    avg_error = total_loss / float(len(test_data))

    return avg_error
