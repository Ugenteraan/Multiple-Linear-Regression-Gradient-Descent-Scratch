import numpy as np
import settings
from load_data import *

np.random.seed(33)

x_length = train_x.shape[1] #number of columns (because each data point is represented as a row in the matrix)

#shape of weight : (5,1) - for this particular dataset
#normally in mathematics, weight would be a matrix 
weight = np.random.rand(x_length, 1) #initialize the weight(theta) vector randomly from normal distribution with the size of the input vector

def find_y_hat(data_point,weight):
	'''
	Parameters :
		data_point 	: Batches of rows in the training set matrix
		weight 		: Weight vector

	Process : matrix (data points) multiplies vector (weight) 

	Output : prediction of the model, y_hat or h(X)
	'''
	return np.dot(data_point, weight) 


def calc_partial_derivative(y_hat, y, x, j,m):
	'''	
	Parameters:	
		y_hat 	: prediction of the model, h(X)
		y 		: label
		x 	 	: input dataset
		j		: weight index
		m 		: total number of data point

	Process : Calculates the partial derivative of the loss with respect to the given weight
			: (h(X) - y)X_j for each data point
			: average the total by 1/m

	Output  : Calculated partial derivative of the loss with respect to the given weight
	'''
	par_der = 0

	for i in range(m):

		par_der += (y_hat[i][0] - y[i][0])*x[i][j] 

	return (1/m)*par_der # (1/m)(sum_{i=1}^{m})(h(x^(i)) - y^(i))x_j^(i)




def train():
	'''
	Process : training of the model
	'''
	for epoch in range(settings.epoch): #One epoch is defined as ONE ROUND of forward and backpropagation of EACH training data point (each row in the matrix)

		loss = 0

		for index_first in range(0, num_of_training_examples, settings.batch_size): #iterate through all the data points in the training set

			index_last = index_first + settings.batch_size #index_first - index_last = batch size

			index_last = None if index_last > num_of_training_examples else index_last #if index_last > total no. of data points, it will be set to None

			y_hat = find_y_hat(train_x[index_first:index_last], weight) #predicted output

			m = y_hat.shape[0] # number of data points

			#initialize	
			sum_of_squares = 0

			#sum of squared errors
			for i in range(m): #sigma^{m}_{i=0} (y_hat^(i) - y^(i))^2

				sum_of_squares += (y_hat[i][0] - train_y[index_first:index_last][i][0])**2 

			loss += (1/(2 * m))*sum_of_squares #MSE (1/2m)(sigma^{m}_{i=0} (y_hat^(i) - y^(i))^2)

			#backpropagation to update the weights
			for j in range(weight.shape[0]): #iterate through all the weights

				partial_derivative = calc_partial_derivative(y_hat, train_y[index_first:index_last], train_x[index_first:index_last], j,m) #par_der(Loss)/par_der(weight[j])

				new_weight_value = weight[j] - (settings.learning_rate * partial_derivative) #weight[j] - (learning_rate * par_der(Loss)/par_der(weight[j]))

				weight[j] = new_weight_value #update the weight with the new value


		print("Epoch : ", epoch)
		print("Training Loss : ", loss) #total loss of the whole epoch


def test():
	'''
	Process : uses the weights to perform feed forward on 'unseen' data points
	'''
	loss = 0 #initialize

	for index_first in range(0, num_of_testing_examples, settings.batch_size):

		index_last = index_first + settings.batch_size #index_first - index_last = batch size

		index_last = None if index_last > num_of_training_examples else index_last #if index_last > total no. of data points, it will be set to None

		y_hat = find_y_hat(test_x[index_first:index_last], weight) #predicted output

		m = y_hat.shape[0] #number of data points

		#initialize
		sum_of_squares = 0

		#sum of squared errors
		for i in range(m): #sigma^{m}_{i=0} (y_hat^(i) - y^(i))^2

			sum_of_squares += (y_hat[i][0] - test_y[index_first:index_last][i][0])**2 

		loss += (1/(2 * m))*sum_of_squares #MSE (1/2m)(sigma^{m}_{i=0} (y_hat^(i) - y^(i))^2)

	return loss #loss of the entire testing dataset




initial_test_loss = test() #run testing before the training
train() #training
final_test_loss = test() #run testing after the training

print("Initial test loss : ", initial_test_loss)
print("Final test loss : ", final_test_loss)
