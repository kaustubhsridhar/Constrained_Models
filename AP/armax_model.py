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

class armax_model(nn.Module):
    def __init__(self, no_of_glucose_inputs = 10, no_of_insulin_inputs = 10, no_of_meal_inputs = 10):

        output_dim = 1
        super(armax_model, self).__init__()

        self.glucose_weights = nn.Parameter(torch.randn((no_of_glucose_inputs, output_dim)))
        self.insulin_weights = nn.Parameter(torch.randn((no_of_insulin_inputs, output_dim)))
        self.meal_weights = nn.Parameter(torch.randn((no_of_meal_inputs, output_dim)))

        self.bias = nn.Parameter(torch.randn((output_dim,)))


    def forward(self, glucose, insulin, meal):
        glucose_term = torch.matmul(glucose, self.glucose_weights)
        # Enforcing insulin having negative effects
        insulin_term = torch.matmul(insulin, -torch.abs(self.insulin_weights))
        meal_term = torch.matmul(meal, self.meal_weights)

        y = glucose_term + insulin_term + meal_term
        return y


def fit_model(train_data, destination_dir = "./networks/"):

    episode_count = 100
    batch_size = 10

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    filename = os.path.join(destination_dir, "diabetes_armax_model.nt")
    print("Length = ", len(train_data))
    print("Input length = ", len(train_data[0]))
    model = armax_model(len(train_data[0][0]), len(train_data[0][1]), len(train_data[0][2]))
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
                glucose_input = np.array(train_data[entry][0][:], dtype = np.float32)
                glucose_input = torch.from_numpy(glucose_input).to(device)

                insulin_input = np.array(train_data[entry][1][:], dtype = np.float32)
                insulin_input = torch.from_numpy(insulin_input).to(device)

                meal_input = np.array(train_data[entry][2][:], dtype = np.float32)
                meal_input = torch.from_numpy(meal_input).to(device)


                target = np.array([train_data[entry][3]], dtype = np.float32)
                target = torch.from_numpy(target).to(device)
                prediction = model.forward(glucose_input, insulin_input, meal_input)

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
        filename = os.path.join("./networks/", "diabetes_armax_model.nt")
        model = torch.load(filename)


    total_loss = 0.0
    for index in range(len(test_data)):

        glucose_input = np.array(test_data[index][0][:], dtype = np.float32)
        glucose_input = torch.from_numpy(glucose_input)

        insulin_input = np.array(test_data[index][1][:], dtype = np.float32)
        insulin_input = torch.from_numpy(insulin_input)

        meal_input = np.array(test_data[index][2][:], dtype = np.float32)
        meal_input = torch.from_numpy(meal_input)

        target = test_data[index][3]

        prediction = model.forward(glucose_input, insulin_input, meal_input)
        prediction = prediction.cpu().detach().numpy()[0]

        loss = abs( prediction - target)
        total_loss += loss


    avg_error = total_loss / float(len(test_data))

    return avg_error

def extract_state_dict():

    filename = os.path.join("./networks/", "diabetes_armax_model.nt")
    model = torch.load(filename)
    torch.save(model.state_dict(), "./networks/diabetes_armax_model_state_dict.nt")

    