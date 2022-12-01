#
# Code written by : Souradeep Dutta,
#  duttaso@seas.upenn.edu, souradeep.dutta@colorado.edu
# Website : https://sites.google.com/site/duttasouradeep39/
#

import os
import csv
import random
import numpy as np
import sys
import shutil
import cv2
import json

def create_regression_data(glucose_src, insulin_src, split_ratio, history_length = 10, prediction_horizon = 5):

    assert split_ratio > 0.0 and split_ratio <= 1.0

    random.seed(0)

    assert glucose_src.find('csv') > -1
    assert insulin_src.find('csv') > -1

    glucose_matrix = []
    with open(glucose_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            glucose_matrix.append(trace)

    insulin_matrix = []
    with open(insulin_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            insulin_matrix.append(trace)

    assert len(glucose_matrix) > 0
    assert len(glucose_matrix) == len(insulin_matrix)
    assert len(glucose_matrix[0]) == len(insulin_matrix[0])


    all_data = []
    no_of_traces = len(glucose_matrix)

    for trace_index in range(no_of_traces):
        trace_length = len(glucose_matrix[0])
        assert trace_length == len(insulin_matrix[0])
        for start_time in range(history_length, trace_length - (prediction_horizon), 1):
            end_time = start_time + prediction_horizon

            glucose_slice = glucose_matrix[trace_index][start_time - history_length : start_time]
            insulin_slice = insulin_matrix[trace_index][start_time - history_length : start_time]
            glucose_prediction = glucose_matrix[trace_index][end_time]

            training_pair = ( glucose_slice + insulin_slice, glucose_prediction)
            all_data.append(training_pair)

    random.shuffle(all_data)

    cut_point = int(float(split_ratio) * float(len(all_data)))
    train_data = all_data[: cut_point]
    test_data = all_data[cut_point:]


    return train_data, test_data


def create_regression_data_with_meals(glucose_src, insulin_src, meals_src, split_ratio, history_length = 10, prediction_horizon = 5):

    assert split_ratio > 0.0 and split_ratio <= 1.0

    random.seed(0)

    assert glucose_src.find('csv') > -1
    assert insulin_src.find('csv') > -1
    assert meals_src.find('csv') > -1

    glucose_matrix = []
    with open(glucose_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            glucose_matrix.append(trace)

    insulin_matrix = []
    with open(insulin_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            insulin_matrix.append(trace)


    meal_matrix = []
    with open(meals_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            meal_matrix.append(trace)


    assert len(glucose_matrix) > 0
    assert len(glucose_matrix) == len(insulin_matrix)
    assert len(glucose_matrix) == len(meal_matrix)
    assert len(glucose_matrix[0]) == len(insulin_matrix[0])
    assert len(glucose_matrix[0]) == len(meal_matrix[0])


    all_data = []
    no_of_traces = len(glucose_matrix)

    for trace_index in range(no_of_traces):
        trace_length = len(glucose_matrix[0])
        assert trace_length == len(insulin_matrix[0])
        for start_time in range(history_length, trace_length - (prediction_horizon), 1):
            end_time = start_time + prediction_horizon

            glucose_slice = glucose_matrix[trace_index][start_time - history_length : start_time]
            insulin_slice = insulin_matrix[trace_index][start_time - history_length : start_time]
            meal_slice = meal_matrix[trace_index][start_time - history_length : start_time]

            glucose_prediction = glucose_matrix[trace_index][end_time]

            training_pair = (glucose_slice, insulin_slice, meal_slice, glucose_prediction)
            all_data.append(training_pair)

    random.shuffle(all_data)

    cut_point = int(float(split_ratio) * float(len(all_data)))
    train_data = all_data[: cut_point]
    test_data = all_data[cut_point:]


    return train_data, test_data

def create_regression_data_with_meals_reformatted(glucose_src, insulin_src, meals_src, glucose_normalizer=None, insulin_normalizer=None, meal_normalizer=None, history_length = 10, prediction_horizon = 5):

    random.seed(0)

    assert glucose_src.find('csv') > -1
    assert insulin_src.find('csv') > -1
    assert meals_src.find('csv') > -1

    glucose_matrix = []
    with open(glucose_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            glucose_matrix.append(trace)

    insulin_matrix = []
    with open(insulin_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            insulin_matrix.append(trace)


    meal_matrix = []
    with open(meals_src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            trace = []
            for each_entry in row :
                trace.append(float(each_entry))
            meal_matrix.append(trace)

    assert len(glucose_matrix) > 0
    assert len(glucose_matrix) == len(insulin_matrix)
    assert len(glucose_matrix) == len(meal_matrix)
    assert len(glucose_matrix[0]) == len(insulin_matrix[0])
    assert len(glucose_matrix[0]) == len(meal_matrix[0])

    states, controls, next_states, traj_starts = [], [], [], []
    no_of_traces = len(glucose_matrix)

    if (glucose_normalizer is None):
        glucose_normalizer = np.max(np.abs(np.array(glucose_matrix)))
    if (insulin_normalizer is None):
        insulin_normalizer = np.max(np.abs(np.array(insulin_matrix)))
    if (meal_normalizer is None):
        meal_normalizer = np.max(np.abs(np.array(meal_matrix)))

    for trace_index in range(no_of_traces):
        trace_length = len(glucose_matrix[0])
        assert trace_length == len(insulin_matrix[0])
        for start_time in range(history_length, trace_length - (prediction_horizon), 1):
            end_time = start_time + prediction_horizon

            glucose_slice = glucose_matrix[trace_index][start_time - history_length : start_time]
            insulin_slice = insulin_matrix[trace_index][start_time - history_length : start_time]
            meal_slice = meal_matrix[trace_index][start_time - history_length : start_time]

            glucose_prediction = glucose_matrix[trace_index][end_time]

            # NEW
            glucose_slice = list(np.array(glucose_slice)/glucose_normalizer)
            insulin_slice = list(np.array(insulin_slice)/insulin_normalizer)
            meal_slice = list(np.array(meal_slice)/meal_normalizer)
            glucose_prediction = glucose_prediction / glucose_normalizer

            is_traj_starting = [1] if start_time == history_length else [0]
            traj_starts.append(is_traj_starting)
            states.append(glucose_slice+insulin_slice+meal_slice)
            controls.append([insulin_slice[-1]])
            next_states.append([glucose_prediction])

    print(f'{len(glucose_slice)=}, {len(insulin_slice)=}, {len(meal_slice)=}')

    return (np.array(states), np.array(controls), np.array(next_states), np.array(traj_starts), 
            glucose_normalizer, insulin_normalizer, meal_normalizer)
