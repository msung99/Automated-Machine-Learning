#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
import numpy as np

# Load the csv file at the given path into the pandas DataFrame and return the DataFrame
def load_dataset(dataset_path):
	#To-Do: Implement this function

# For the given DataFrame, return the following statistical analysis results in order
# Number of features / Number of data for class 0 / Number of data for class 1
def dataset_stat(dataset_df):	
	#To-Do: Implement this function

# Splitting the given DataFrame and return train data, test data, train label, and test label in order
# You must split the data using the given test size
def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function

# Using the given train dataset, train the decision tree model
# You can implement with "default arguments"
# After the training, evaluate the performances of the model using the given test dataset
# Return three performance metrics (accuracy, precision, recall) in order
def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

# Using the given train dataset, train the random forest model
# You can implement with default arguments
# After the training, evaluate the performances of the model using the given test dataset
# Return three performance metrics (accuracy, precision, recall) in order
def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

# Using the given train dataset, train the pipeline consists of a standard scaler and SVM
# You can implement with default arguments
# After the training, evaluate the performances of the model using the given test dataset
# Return three performance metrics (accuracy, precision, recall) in order
def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)