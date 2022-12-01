from data_manager import *
from model import *
import armax_model

if __name__ == "__main__" :
    glucose_src = "./insulin_matlab/Glucose_data.csv"
    insulin_src = "./insulin_matlab/Insulin_data.csv"
    meals_src = "./insulin_matlab/meal_data.csv"

    mode = "train_armax_model"

    split_ratio = 0.8

    if mode == "test_vanilla_nn" :
        train_data, test_data = create_regression_data(glucose_src, insulin_src, split_ratio)

        # model = fit_model(train_data)

        accuracy = test_model(test_data)

        print("Average prediction error = ", accuracy)

    elif mode == "train_armax_model" :

        train_data, test_data = create_regression_data_with_meals(glucose_src, insulin_src, meals_src, split_ratio)

        model = armax_model.fit_model(train_data)

        accuracy = armax_model.test_model(test_data)

        print("Average prediction error = ", accuracy)
    
    elif mode == "test_armax_model" :

        train_data, test_data = create_regression_data_with_meals(glucose_src, insulin_src, meals_src, split_ratio)

        accuracy = armax_model.test_model(test_data)

        print("Average prediction error = ", accuracy)
