# Project <Titanic Machine Learning>
# Made by Daniil Khmelnytskyi
# 04.04.2022

# Imports 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def percantage_of_alive(train_data):
    """
        Func calculates percantage of alive women and men
    """
    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    rate_women = sum(women) / len(women)

    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    rate_men = sum(men) / len(men)

    print(f"% of women who survived: {rate_women}")
    print(f"% of men who survived: {rate_men}")


def initialize_forest(X, y):
    """
        Func initializes `random forest` and train it
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)

    return model
    

def predict(model, X_test, train_data_ids):
    """
        Func makes prediction and saves submission
    """
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': train_data_ids, 'Survived': predictions})
    output.to_csv('result/submission.csv', index=False)

    print("Submission was successfully saved")


def calc_accuracy():
    """
        Func checks accuracy of prediction
    """
    y_data = pd.read_csv("result/submission.csv")
    right_data = pd.read_csv("data/gender_submission.csv")

    right_data_refactored = dict(list(zip(list(right_data['PassengerId']), list(right_data['Survived']))))

    amount_of_all = len(right_data_refactored)
    counter_of_right = 0

    for i in range(len(y_data)):
        if right_data_refactored[y_data['PassengerId'][i]] == y_data['Survived'][i]:
            counter_of_right += 1

    print(f"Amount of all test records: {amount_of_all}")
    print(f"Counter of right predicted records: {counter_of_right}")

    accuracy = counter_of_right / amount_of_all * 100
    print(f'Accuracy is: {accuracy}')


if __name__ == "__main__":
    # Get data to train && test
    train_data = pd.read_csv("data/train.csv") # read training data

    test_data = pd.read_csv("data/test.csv") # read testing data

    percantage_of_alive(train_data) # percantage of alive men and women

    features = ["Pclass", "Sex", "SibSp", "Parch"] # columns to extract from csv

    X = pd.get_dummies(train_data[features]) # converts categorical data into indicator variables (sex => male && female)
    y = train_data["Survived"] # extract column 'Survived'

    model_forest = initialize_forest(X, y) # initialize 'random forest' and train it

    X_test = pd.get_dummies(test_data[features]) # converts categorical data into indicator variables (sex => male && female)

    predict(model_forest, X_test, test_data.PassengerId) # predicts on test_data with trained model

    calc_accuracy() # calculates accuracy of model