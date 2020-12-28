import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split



def Adj_mat_with_euclidean(data_x, sig=1, threshold=0.5):
	# calculate adjancy matrix with euclidean distance

	# Input
	# data_x: ndarray N X F (N: # of nodes, F: # of features)

	# Output: Adj matrix

	dist = euclidean_distances(data_x)
	simil = np.exp(-dist/(sig**2))
	selected = np.array(np.where(simil>=threshold))

	Adj = np.zeros((len(data_x), len(data_x)))
	Adj[selected] = 1
	return Adj


def get_k_mask(x_data):

	nonzero_idx = np.argwhere(np.sum(x_data, axis=0)!=0)

	Xs = np.zeros((len(nonzero_idx)+1, x_data.shape[0], x_data.shape[1])) 
	Xs[0,:,:] = x_data

	for idx, val in enumerate(nonzero_idx): 
	    temp_data = np.copy(x_data) 
	    temp_data[:,val] = 0 
	    Xs[idx+1,:,:] = temp_data

	return Xs


def load_bench_iris(test_size=0.3, shuffle=True, random_state=1004):
	raw_iris = datasets.load_iris()
	Iris_x = raw_iris.data
	Iris_y = np.eye(3)[raw_iris.target]
	Adj = Adj_mat_with_euclidean(iris_x)
	
	# split data to train/test
	X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(Iris_x, Iris_y, range(len(Irix_x)), 
		test_size=test_size, shuffle=shuffle, random_state=random_state)
	y_train = Iris_y
	y_test = Iris_y
	y_train[test_index] = 0
	y_test[train_index] = 0

	# boolean type of index for model train
	index_train = np.sum(y_train, axis=1)
	index_train = train_index == 1


	Xs = get_k_mask(Iris_x)
	Y_train = y_train.astype(np.float32)

	return Xs, Adj, Y_train, y_test, idx_train










