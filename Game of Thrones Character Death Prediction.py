"""
File: game of thrones_decision tree.py
Name: 513716004 陳映廷(Gloria)
---------------------------
This file shows how to use pandas and sklearn
packages to build a decision tree.
"""

import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import pydotplus

# Constants - filenames for data set
TRAIN_FILE = 'game of thrones_data/character-deaths.csv'


def main():
    # Data preprocessing
    data = data_preprocess(TRAIN_FILE)

    # Inspecting if there is any NaN data
    # print(data.isna().sum())

    # Extract true labels
    y = data['Death Year']
    # print("This is true labels", y)

    # Extract features
    x = data.iloc[:, 4:]
    # print("This is features", x)

    # Splitting the data into training data and testing data (75% Training data, 25% Testing data)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75)

    # Construct Tree
    d_tree = tree.DecisionTreeClassifier(max_depth=20, random_state=0)
    d_tree_classifier = d_tree.fit(x_train, y_train)

    # Test Data
    prediction = d_tree_classifier.predict(x_test)

    # Calculate the Precision, Recall, Accuracy within Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, prediction)
    class_report = metrics.classification_report(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    recall = metrics.recall_score(y_test, prediction)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print('Confusion Matrix:\n', confusion_matrix)
    print(class_report)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Accuracy:', accuracy)

    # Output decision tree for PNG
    dot_data = tree.export_graphviz(d_tree_classifier)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('decision_tree.png')

    print('Output decision tree.png successfully.')


def data_preprocess(filename):
    """
    : param filename: str, the csv file to be read into by pd
    -----------------------------------------------
    This function reads in data by pd, changing string data to float,
    and finally tackling missing data showing as NaN on pandas
    """
    # Read in data as a column based DataFrame
    data = pd.read_csv(filename)

    # Replacing NaN with 0 --> data cleaning
    data['Book Intro Chapter'] = data['Book Intro Chapter'].fillna(0)

    # Replacing NaN with 0 --> means the character is alive.
    data['Death Year'] = data['Death Year'].fillna(0)
    data['Book of Death'] = data['Book of Death'].fillna(0)
    data['Death Chapter'] = data['Death Chapter'].fillna(0)

    # Changing numbers with 1 --> means the character is dead.
    data.loc[data['Death Year'] != 0, 'Death Year'] = 1
    data.loc[data['Book of Death'] != 0, 'Book of Death'] = 1
    data.loc[data['Death Chapter'] != 0, 'Death Chapter'] = 1

    # Changing 'Allegiances' to dummies
    data_dummies = pd.get_dummies(data, columns=['Allegiances'])

    return data_dummies


if __name__ == '__main__':
    main()
