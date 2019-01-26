import numpy as np 
import settings

#lists to temporarily hold the readings from the file
x_list = list()
y_list = list()


fin = open(settings.dataset_path)

for line in fin: #read the file line by line

	line = line.strip().split(',') #strip to remove '\n' and split to separate the values from each column
	
	#exception handling for header rows
	try:
		x = [1.0] + [float(s) for s in line[:4]] #the values in first 4 columns will be converted to float, 1 is added for bias calculation later on
		y = [float(line[-1])] #the value in the last column will be converted to float
	
	except ValueError: #header rows can't be converted to float
		continue

	x_list.append(x) 
	y_list.append(y)


num_of_training_examples = int(len(y_list) * (1 - settings.test_training_ratio))
num_of_testing_examples = len(y_list) - num_of_training_examples

#numpy arrays
#each data point is represented as a row in the matrix (normally we would represent them as columns in mathematics)
train_x = np.asarray(x_list[:num_of_training_examples], dtype=np.float) #shape (7654,5) - for this particular dataset
train_y = np.asarray(y_list[:num_of_training_examples], dtype=np.float) #shape (7654,1) - for this particular dataset

test_x = np.asarray(x_list[num_of_training_examples:], dtype=np.float)  #shape (1914, 5) - for this particular dataset
test_y = np.asarray(y_list[num_of_training_examples:], dtype=np.float)  #shape (1914, 1) - for this particular dataset
