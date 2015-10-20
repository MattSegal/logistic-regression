"""
Logistic Regression Module
see readme for details
"""

import numpy as np
import matplotlib.pyplot as plt

# gradient descent parameters
LEARNING_RATE	= 3 
NUM_ITERS	= 1000


# =============================================================================== #

def log_reg(training_arr,result_arr,num_examples,num_features,plot_convergence=False):
    """
    training array should be formatted using format_array
    training_arr is expected to come with training_arr bias entry already added 
	ie training_arr has dim (examples,features+1)
	most easily done with normalize or add_bias functions
    """
    # prepare training array
    training_arr = format_array(training_arr,num_examples,num_features)
    norm_info = normalize(training_arr)
    training_arr = apply_norm(training_arr,norm_info)

    # check inputs
    assert (num_examples,num_features+1) == training_arr.shape,'training array wrong shape'
    assert num_examples == result_arr.size,'result array wrong size'

    # use gradient descent to find weights
    initial_weights   = np.zeros(num_features+1)
    weights = gradient_descent(training_arr, result_arr, initial_weights, plot_convergence)
    return weights,norm_info

def gradient_descent(training_arr, result_arr, weights,plot_convergence = False):
    """
    performs gradient descent to minimize cost function of the weight values
    returns weights that are (hopefully) a good fit to data
    """
    # Initialize some useful values
    (num_examples,num_features) = training_arr.shape
    cost_history = np.zeros(NUM_ITERS)
    cost_gradient = np.zeros(num_features)

    for itr in range(NUM_ITERS):
        (cost,cost_gradient) = basic_cost_function(weights,training_arr,result_arr)
        weights = weights - LEARNING_RATE * cost_gradient;
        cost_history[itr] = cost

    if plot_convergence:
        plt.plot(cost_history)
        plt.ylabel('Cost Function')
        plt.xlabel('Number of Iterations')
        plt.show()

    return weights

def basic_cost_function(weights,training_arr,result_arr):
    """
    basic cost function
    no regularization
    in future - implement more advanced techniques
    """
    num_exam = float(len(result_arr))	# number of training examples
    num_feat = weights.size		# number of parameters/features (includes bias term)

    # output variables
    cost = 0
    cost_grad = np.zeros(num_feat)

    # hypothesis result of input data and weights
    lin_hyp = np.dot(training_arr,weights)
    hyp	    = sigmoid(lin_hyp)
    
    # gradient of cost function
    for feature in range(num_feat):
	cost_grad[feature] =  ((1/num_exam)*(hyp - result_arr)*training_arr[:,feature] ).sum()

    # cost function value
    positive_results = - result_arr*np.log(hyp)	    
    negative_results = - (1-result_arr)*np.log(1-hyp)    
    cost = ( (1/num_exam)*(positive_results + negative_results) ).sum()

    return cost , cost_grad


   

# =============================================================================== #

def sigmoid(z):
    """
    sigmoid aka logistic function
    """
    return 1.0/(1.0+np.exp(-z))

# =============================================================================== #

def hypothesis(input_arr,weights,norm_info=None):
    """
    input_arr can be mxn array, float, or integer (n==1)
    weights are an array of size n+1
    result is a value between 0 and 1
    """
    num_features = weights.size - 1
    # format and normalzie input array
    input_arr = format_array(input_arr,1,num_features)
    if norm_info == None:
	input_arr = add_bias(input_arr)
    else:
	input_arr = apply_norm(input_arr,norm_info)
    # calculate result
    linear_result = np.dot(input_arr,weights)
    result = sigmoid(linear_result)
    assert result[0] >= 0 and result[0] <= 1,'result %.1f is outside of (0,1) bound' % result[0]
    assert result.ndim == 1 and result.size == 1,'result has incorrect dimensions'
    return result[0]

# =============================================================================== #

def normalize(input_arr, norm_type = 'gaussian'):
    """
    input_arr is of dimensions (m,n) with m examples and n features
    normalizes input_arr and returns norm_info for each feature
	for gaussian norm_info contains feature mu (mean) and sigma (stdev)
	for minmax norm_info contains feature min and max values
    function assumes array is already *formatted*
    returns norm_info
    """
    num_features   = input_arr.shape[1]

    if norm_type == 'gaussian':
	mu =    np.zeros(num_features+1)
	sigma = np.zeros(num_features+1)
	mu[0]           = 0 # we don't normalize bias
	sigma[0]        = 1 

	for feat in range(1,num_features+1): # can probably be vectorized
	    mu[feat]    = np.mean(input_arr[:,feat-1],axis=0)
	    sigma[feat] = np.std(input_arr[:,feat-1],axis=0)
	norm_info = {'type':'gaussian','mu':mu,'sigma':sigma}

    elif norm_type == 'minmax':
	min_val = np.zeros(num_features+1)
	max_val = np.zeros(num_features+1)
	min_val[0]      = 0 # we don't normalize bias         
	max_val[0]      = 1 
	for feat in range(1,num_features+1): # can probably be vectorized
	    min_val[feat] = np.min(input_arr[:,feat-1],axis=0)
	    max_val[feat] = np.max(input_arr[:,feat-1],axis=0)
	norm_info = {'type':'minmax','min_val':min_val,'max_val':max_val}
    else:
	raise ValueError('normalization type invalid')

    return norm_info

def apply_norm(input_arr,norm_info):
    """
    normalize input data based on norm_info
    and add bias term (1) to first column
    takes mxn dimension numpy array, integer or float as input
	m - number of examples
	n - number of features

    array must be pre-formatted by format_array
    array must not yet contain bias term

    N.B consider vectorizing this code - may as well
    """
    (examples,features) = input_arr.shape
    input_norm = add_bias(input_arr)

    if norm_info['type'] == 'gaussian':
	mu = norm_info['mu']
	sigma = norm_info['sigma']
	for feat in range(1,features+1):
	    sigma_val = sigma[feat] + 1*(sigma[feat]==0)
	    input_norm[:,feat] = (input_arr[:,feat-1] - mu[feat]) / sigma_val

    elif norm_info['type'] == 'minmax':
	min_val = norm_info['min_val']
	max_val = norm_info['max_val']
	for feat in range(1,features+1):
		input_norm[:,feat] = (input_arr[:,feat-1] - min_val[feat]) / max_val[feat]
    else:
	raise ValueError('normalization type invalid')
    return input_norm

def format_array(input_arr,num_examples,num_features):
    """
    reformats input array to an (m x n) ndarray where
	m is the number of an examples
	n is the number of features
	a 1x1 array will be in format [[val]]
	ie, array always has 2 dimensions
    input cases:
	int/float
	ndarray 0x0, 1x0,  1x1, mx1, 1xn, mxn
    """
    input_arr = np.array(input_arr,dtype=np.float) # ensure numpy array
    
    if num_features == 1 and num_examples == 1:
	assert input_arr.ndim <=2,'too many dimensions'
	assert input_arr.size == 1,'too many elements'
	if input_arr.ndim == 0:
	    output_arr = np.array([[input_arr]])
	elif input_arr.ndim == 1:
	    output_arr = np.array([input_arr])
	else: #input_arr.ndim == 2
	    output_arr = input_arr
    else:
	if num_features == 1:
	    assert num_examples == input_arr.size,'incorrect num_examples'
	    new_shape = (num_examples,num_features)
	    output_arr = input_arr.reshape(new_shape)
	elif num_examples == 1:
	    assert num_features == input_arr.size,'incorrect num_features'
	    new_shape = (num_examples,num_features)
	    output_arr = input_arr.reshape(new_shape)
	else: # (m x n)
	    assert num_examples*num_features == input_arr.size,'incorrect number of elements'
	    (rows,cols) = input_arr.shape
	    assert rows == num_examples and cols == num_features,'incorrect dimensions'
	    output_arr = input_arr
    return output_arr
    
def add_bias(input_arr):
    """
    adds bias term (1) to first column a *formatted* array
    """
    (num_examples,num_features) = input_arr.shape
    output_arr  = np.ones((num_examples,num_features+1))
    output_arr[:,1:] = input_arr
    return output_arr

# =============================================================================== #

def plot_one_dim_reg(training_arr,result_arr,norm_info,weights,max_val,min_val):
    """
    plots results of 1d logistic regression over training set
    training array is raw (not normalized)
    """
    hyp_x = np.linspace(min_val, max_val, num=100)
    hyp_y = np.zeros(hyp_x.size)
    for idx in range(hyp_y.size):
	hyp_y[idx] = hypothesis(hyp_x[idx],weights,norm_info)

    plt.plot(training_arr,result_arr,'bo',ms=10) # plot training set
    plt.plot(hyp_x,hyp_y) # plot hypothesis function
    plt.ylim(0,1.05)
    plt.xlim(min_val,max_val+10)
    plt.ylabel('Result')
    plt.xlabel('Training Values')
    plt.show()

def plot_reg(weights,x_min,x_max):
    """
    plots results of a contrived logistic regression
    weights given as input
    useful for conceptual understanding
    """
    hyp_x = np.linspace(x_min, x_max, num=1000)
    X = np.ones((hyp_x.size,2))
    X[:,1] = hyp_x
    hyp_y = sigmoid( np.dot(X,weights) )

    plt.plot(hyp_x,hyp_y) # plot hypothesis function
    plt.ylabel('Output')
    plt.xlabel('Input')
    plt.show()

def plot_decision_boundary(training_arr,result_arr,norm_info,weights):
    """
    plots decision boundary
    decision boundary is where linear hypothesis is 0
    no current plans to incorporate polynomial hypotheses
    """
    # not implemented
    pass

def plot_cost_function_contour():
    """
    this might be cool and helpful for insight
    might be a bitch to write tho
    """
    # not implemented
    pass

def plot_normalization():
    """
    somehow visualizes effects of normalization
    might be cool
    """
    pass





    
