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


def fit_model(all_states, all_controls, all_next_states, glucose_normalizer, insulin_normalizer, meal_normalizer, filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    episode_count = 100
    batch_size = 64

    model = armax_model().to(device)
    learning_rate = 1e-3
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    all_train_states = np.concatenate((all_states['main']['train'], all_states['omega']['train']), axis=0)
    all_train_controls = np.concatenate((all_controls['main']['train'], all_controls['omega']['train']), axis=0)
    all_train_next_states = np.concatenate((all_next_states['main']['train'], all_next_states['omega']['train']), axis=0)
    T = 10

    num_datapoints = len(all_train_states)
    for episode in range(episode_count):
        number_of_batches = math.ceil(float(num_datapoints) / float(batch_size))

        total_loss = 0.0

        for batch_index in range(number_of_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, num_datapoints)
            loss_list = []
            optimizer.zero_grad()

            # for entry in range(start, end):
            #     glucose_input = all_train_states[entry, :T] * glucose_normalizer
            #     glucose_input = torch.from_numpy(glucose_input).float().to(device)
            #     insulin_input = all_train_states[entry, T:2*T] * insulin_normalizer
            #     insulin_input = torch.from_numpy(insulin_input).float().to(device)
            #     meal_input = all_train_states[entry, 2*T:3*T] * meal_normalizer
            #     meal_input = torch.from_numpy(meal_input).float().to(device)
            #     target = all_train_next_states[entry, :] * glucose_normalizer
            #     target = torch.from_numpy(target).float().to(device)
            #     prediction = model.forward(glucose_input, insulin_input, meal_input)
            #     loss = criterion(prediction, target).to(device)
            #     loss_list.append(loss)
            #     # Computing loss for tracking
            #     total_loss += loss.item()

            glucose_input = all_train_states[start:end, :T] * glucose_normalizer
            glucose_input = torch.from_numpy(glucose_input).float().to(device)

            insulin_input = all_train_states[start:end, T:2*T] * insulin_normalizer
            insulin_input = torch.from_numpy(insulin_input).float().to(device)

            meal_input = all_train_states[start:end, 2*T:3*T] * meal_normalizer
            meal_input = torch.from_numpy(meal_input).float().to(device)

            target = all_train_next_states[start:end, :] * glucose_normalizer
            target = torch.from_numpy(target).float().to(device)
            
            prediction = model.forward(glucose_input, insulin_input, meal_input)

            loss = criterion(prediction, target).to(device)

            # Computing loss for tracking
            total_loss += loss.item() * (end-start)

            loss.backward()
            optimizer.step()


        print("At episode - ", episode, " avg loss computed - ", total_loss / float(len(all_train_states)))
        if (episode+1)%10 == 0 or episode == 0:
            test_loss = test_model(model, all_states, all_controls, all_next_states, glucose_normalizer, insulin_normalizer, meal_normalizer, T)
            print("At episode - ", episode, " avg test loss computed - ", test_loss )

        torch.save(model.state_dict(), filename)

    return model


def test_model(model, all_states, all_controls, all_next_states, glucose_normalizer, insulin_normalizer, meal_normalizer, T):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_test_states = np.concatenate((all_states['main']['test'], all_states['omega']['test']), axis=0)
    all_test_controls = np.concatenate((all_controls['main']['test'], all_controls['omega']['test']), axis=0)
    all_test_next_states = np.concatenate((all_next_states['main']['test'], all_next_states['omega']['test']), axis=0)

    total_loss = 0.0
    for index in range(len(all_test_states)):

        glucose_input = all_test_states[index, :T] * glucose_normalizer
        glucose_input = torch.from_numpy(glucose_input).float().to(device)

        insulin_input = all_test_states[index, T:2*T] * insulin_normalizer
        insulin_input = torch.from_numpy(insulin_input).float().to(device)

        meal_input = all_test_states[index, 2*T:3*T] * meal_normalizer
        meal_input = torch.from_numpy(meal_input).float().to(device)

        target = all_test_next_states[index, :] * glucose_normalizer
        target = torch.from_numpy(target).float().to(device)
        
        prediction = model.forward(glucose_input, insulin_input, meal_input)
        prediction = prediction.cpu().detach().numpy()[0]

        loss = abs( prediction - target)**2
        total_loss += loss


    avg_error = total_loss / float(len(all_test_states))

    return avg_error.item()

def extract_state_dict():

    filename = os.path.join("./networks/", "diabetes_armax_model.nt")
    model = torch.load(filename)
    torch.save(model.state_dict(), "./networks/diabetes_armax_model_state_dict.nt")

    