#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
import numpy as npl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score # 정확도 계산
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report

# Load the csv file at the given path into the pandas DataFrame and return the DataFrame
def load_dataset(dataset_path):
    data_df = pd.read_csv(dataset_path)
    return data_df

# For the given DataFrame, return the following statistical analysis results in order
# Number of features / Number of data for class 0 / Number of data for class 1
def dataset_stat(dataset_df):
	freq_zero = 0
	freq_one = 0
	for data in dataset_df.target.values.tolist():
		if(data == 0):
			freq_zero += 1
		elif(data == 1):
			freq_one += 1
	return (dataset_df.shape[1] -1 , freq_zero, freq_one)

# Splitting the given DataFrame and return train data, test data, train label, and test label in order
# You must split the data using the given test size
def split_dataset(dataset_df, testset_size):
	X = dataset_df.drop(columns = "target", axis = 1)
	y = dataset_df["target"]
	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = testset_size)
	return (X_train, X_test, Y_train, Y_test)

# Using the given train dataset, train the decision tree model
# You can implement with "default arguments"
# After the training, evaluate the performances of the model using the given test dataset
# Return three performance metrics (accuracy, precision, recall) in order
def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt_cls = DecisionTreeClassifier()
	#dt_cls.fit(x_train, y_train) # 훈련 진행
	pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier())
	pipe.fit(x_train, y_train)
	acc = accuracy_score(pipe.predict(x_test), y_test) 
	prec = precision_score(pipe.predict(x_test), y_test) 
	rec = recall_score(pipe.predict(x_test), y_test)
	print(confusion_matrix(y_test, pipe.predict(x_test)))
	return (acc, prec, rec)
	#print(accuracy_score(pipe.predict(x_test), y_test))
	#print(precision_score(pipe.predict(x_test), y_test))
	#print(recall_score(pipe.predict(x_test), y_test))
	#print(confusion_matrix(y_test, pipe.predict(x_test)))

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