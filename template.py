#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
import numpy as np

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
#def split_dataset(dataset_df, testset_size):


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)
	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)