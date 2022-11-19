#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/msung99/Automated-Machine-Learning.git

import sys
import pandas as pd
import numpy as npl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score # 정확도 계산
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import precision_score, recall_score

def load_dataset(dataset_path):
    data_df = pd.read_csv(dataset_path)
    return data_df


def dataset_stat(dataset_df):
	freq_zero = 0
	freq_one = 0
	for data in dataset_df.target.values.tolist():
		if(data == 0):
			freq_zero += 1
		elif(data == 1):
			freq_one += 1
	return (dataset_df.shape[1] -1 , freq_zero, freq_one)


def split_dataset(dataset_df, testset_size):
	X = dataset_df.drop(columns = "target", axis = 1)
	y = dataset_df["target"]
	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = testset_size)
	return (X_train, X_test, Y_train, Y_test)


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier())
	pipe.fit(x_train, y_train)
	acc = accuracy_score(pipe.predict(x_test), y_test) 
	prec = precision_score(pipe.predict(x_test), y_test) 
	rec = recall_score(pipe.predict(x_test), y_test)
	return (acc, prec, rec)


def random_forest_train_test(x_train, x_test, y_train, y_test):
	pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
	pipe.fit(x_train, y_train)
	acc = accuracy_score(pipe.predict(x_test), y_test) 
	prec = precision_score(pipe.predict(x_test), y_test) 
	rec = recall_score(pipe.predict(x_test), y_test)
	return (acc, prec, rec)


def svm_train_test(x_train, x_test, y_train, y_test):
	pipe = make_pipeline(StandardScaler(), SVC())
	pipe.fit(x_train, y_train)
	acc = accuracy_score(pipe.predict(x_test), y_test) 
	prec = precision_score(pipe.predict(x_test), y_test) 
	rec = recall_score(pipe.predict(x_test), y_test)
	return (acc, prec, rec)


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
	