"""
tests functions/features of log_reg.py
for use with py.test unit testing framework
"""

from log_reg import *

def test_format_array():
    # 1 feature 1 example
    correct = np.array([[1]])
    # integer
    input_arr = 1
    output = format_array(input_arr,1,1)
    assert output == correct
    # float
    input_arr = 1.0
    output = format_array(input_arr,1,1)
    assert output == correct
    # list
    input_arr = [1]
    output = format_array(input_arr,1,1)
    assert output == correct
    # ndarray 0x0
    input_arr = np.array(1)
    output = format_array(input_arr,1,1)
    assert output == correct
    # ndarray 1x0
    input_arr = np.array([1])
    output = format_array(input_arr,1,1)
    assert output == correct
    # ndarray 1x1
    input_arr = np.array([[1]])
    output = format_array(input_arr,1,1)
    assert output == correct

    # 1 feature multiple examples (mx1)
    correct = np.array([[1],[2],[3],[4]])
    # ndarray mx0
    input_arr = np.array([1,2,3,4])
    output = format_array(input_arr,4,1)
    assert (output == correct).all()
    assert output.shape == correct.shape
    # ndarray 1xm
    input_arr = np.array([[1,2,3,4]])
    output = format_array(input_arr,4,1)
    assert (output == correct).all()
    assert output.shape == correct.shape
    # ndarray mx1
    input_arr =np.array([[1],[2],[3],[4]])
    output = format_array(input_arr,4,1)
    assert (output == correct).all()
    assert output.shape == correct.shape

    # 1 example multiple features (1xn)
    correct = np.array([[1,2,3,4]])
    # ndarray nx0
    input_arr = np.array([1,2,3,4])
    output = format_array(input_arr,1,4)
    assert (output == correct).all()
    assert output.shape == correct.shape
    # ndarray 1xn
    input_arr = np.array([[1,2,3,4]])
    output = format_array(input_arr,1,4)
    assert (output == correct).all()
    assert output.shape == correct.shape
    # ndarray nx1
    input_arr =np.array([[1],[2],[3],[4]])
    output = format_array(input_arr,1,4)
    assert (output == correct).all()
    assert output.shape == correct.shape

    # multiple features and examples (mxn)
    # not sure how to rigorously test this case
    correct = np.array([[1,2],[3,4]])
    input_arr =np.array([[1,2],[3,4]])
    output = format_array(input_arr,2,2)
    assert (output == correct).all()
    assert output.shape == correct.shape

def test_add_bias():
    input_arr = np.array([[1]])
    correct = np.array([[1,1]])
    output = add_bias(input_arr)
    assert (output == correct).all()

    input_arr = np.array([[1],[2],[3],[4]])
    correct = np.array([[1,1],[1,2],[1,3],[1,4]])
    output = add_bias(input_arr)
    assert (output == correct).all()

    input_arr = np.array([[1,2,3,4]])
    correct = np.array([[1,1,2,3,4]])
    output = add_bias(input_arr)
    assert (output == correct).all()

    input_arr = np.array([[1,2],[3,4]])
    correct = np.array([[1,1,2],[1,3,4]])
    output = add_bias(input_arr)
    assert (output == correct).all()

def test_normalize():

    norm_style = 'gaussian'
    input_arr = np.array([[1]])
    norm_info = normalize(input_arr, norm_style)
    mu = norm_info['mu']
    sigma = norm_info['sigma']
    correct_mu = np.array([0,1])
    correct_sigma = np.array([1,0])
    assert (mu == correct_mu).all()
    assert (sigma == correct_sigma).all()
    
    norm_style = 'gaussian'
    input_arr = np.array([[1],[2],[3],[4]])
    norm_info = normalize(input_arr, norm_style)
    mu = norm_info['mu']
    sigma = norm_info['sigma']
    correct_mu = np.array([0,2.5])
    correct_sigma = np.array([1,np.std([1,2,3,4])])
    assert (mu == correct_mu).all()
    assert (sigma == correct_sigma).all()

    norm_style = 'gaussian'
    input_arr = np.array([[1,2,3,4]])
    norm_info = normalize(input_arr, norm_style)
    mu = norm_info['mu']
    sigma = norm_info['sigma']
    correct_mu = np.array([0,1,2,3,4])
    correct_sigma = np.array([1,0,0,0,0])
    assert (mu == correct_mu).all()
    assert (sigma == correct_sigma).all()

    norm_style = 'gaussian'
    input_arr = np.array([[1,2],[3,4]])
    norm_info = normalize(input_arr, norm_style)
    mu = norm_info['mu']
    sigma = norm_info['sigma']
    correct_mu = np.array([0,2,3])
    correct_sigma = np.array([1,np.std([1,3]),np.std([2,4])])
    assert (mu == correct_mu).all()
    assert (sigma == correct_sigma).all()

    norm_style = 'minmax'
    input_arr = np.array([[1]])
    norm_info = normalize(input_arr, norm_style)
    min_val = norm_info['min_val']
    max_val = norm_info['max_val']
    correct_min = np.array([0,1])
    correct_max = np.array([1,1])
    assert (min_val == correct_min).all()
    assert (max_val == correct_max).all()

    norm_style = 'minmax'
    input_arr = np.array([[1],[2],[3],[4]])
    norm_info = normalize(input_arr, norm_style)
    min_val = norm_info['min_val']
    max_val = norm_info['max_val']
    correct_min = np.array([0,1])
    correct_max = np.array([1,4])
    assert (min_val == correct_min).all()
    assert (max_val == correct_max).all()

    norm_style = 'minmax'
    input_arr = np.array([[1,2,3,4]])
    norm_info = normalize(input_arr, norm_style)
    min_val = norm_info['min_val']
    max_val = norm_info['max_val']
    correct_min = np.array([0,1,2,3,4])
    correct_max = np.array([1,1,2,3,4])
    assert (min_val == correct_min).all()
    assert (max_val == correct_max).all()

    norm_style = 'minmax'
    input_arr = np.array([[1,2],[3,4]])
    norm_info = normalize(input_arr, norm_style)
    min_val = norm_info['min_val']
    max_val = norm_info['max_val']
    correct_min = np.array([0,1,2])
    correct_max = np.array([1,3,4])
    assert (min_val == correct_min).all()
    assert (max_val == correct_max).all()


def test_apply_norm():
    # ndarray input
    input_arr = np.array([[1.]])
    norm_style = 'gaussian'
    norm_info = normalize(input_arr,norm_style) # mu 1 sigma 0
    input_norm = apply_norm(input_arr,norm_info)
    # mean is 1 so [1] normalized to 0
    correct = np.array([[1.,0.]])
    assert (input_norm == correct).all()

    input_arr = np.array([[1.],[2.],[3.],[4.]])
    norm_style = 'gaussian'
    norm_info = normalize(input_arr,norm_style)
    input_norm = apply_norm(input_arr,norm_info)
    correct = np.array([[1,1],[1,2],[1,3],[1,4]],dtype=np.float)
    correct_mu = np.array([0,2.5])
    correct_sigma = np.array([1,np.std([1,2,3,4])])
    correct = (correct - correct_mu) / correct_sigma
    assert (input_norm == correct).all()

    input_arr = np.array([[1.,2.],[3.,4.]])
    norm_style = 'gaussian'
    norm_info = normalize(input_arr,norm_style)
    input_norm = apply_norm(input_arr,norm_info)
    correct = np.array([[1,1,2],[1,3,4]],dtype=np.float)
    correct_mu = np.array([0,2,3])
    correct_sigma = np.array([1,np.std([1,3]),np.std([2,4])])
    correct = (correct - correct_mu) / correct_sigma
    assert (input_norm == correct).all()

    input_arr = np.array([[1.]])
    norm_style = 'minmax'
    norm_info = normalize(input_arr,norm_style) # mu 1 sigma 0
    input_norm = apply_norm(input_arr,norm_info)
    correct = np.array([1.,0.])
    assert (input_norm == correct).all()

    input_arr = np.array([[1.],[2.],[3.],[4.]])
    norm_style = 'minmax'
    norm_info = normalize(input_arr,norm_style)
    input_norm = apply_norm(input_arr,norm_info)
    correct = np.array([[1,1],[1,2],[1,3],[1,4]],dtype=np.float)
    correct_min = np.array([0.,1.])
    correct_max = np.array([1.,4.])
    correct = (correct - correct_min) / correct_max
    assert (input_norm == correct).all()

    input_arr = np.array([[1.,2.],[3.,4.]])
    norm_style = 'minmax'
    norm_info = normalize(input_arr,norm_style)
    input_norm = apply_norm(input_arr,norm_info)
    correct = np.array([[1,1,2],[1,3,4]],dtype=np.float)
    correct_min = np.array([0.,1.,2.])
    correct_max = np.array([1.,3.,4.])
    correct = (correct - correct_min) / correct_max
    assert (input_norm == correct).all()

def test_hypothesis():
    """
    check that float, int and ndarray are all accepted as input
    probably should check if it gets the right answer for
    any value but 1...
    """
    weights = np.array([0,1]) # trivial, should get sig(1) as result for input 1
    correct = sigmoid(1)
    # integer input
    input_arr = 1
    result = hypothesis(input_arr,weights)
    assert result == correct
    # float input
    input_arr = 1.0
    result = hypothesis(input_arr,weights)
    assert result == correct
    # ndarray input
    input_arr = np.array([1])
    result = hypothesis(input_arr,weights)
    assert result == correct

def test_basic_cost_function():
    """
    things to test:
	1 feature:
	    positive result - 1 example
	    negative result - 1 example
	    pos + neg result - 2 examples
	2 features:
	    pos + neg result - 2 examples
    """
    # some helper functions
    def hyp(training,weights):
	return sigmoid(np.dot(training,weights))
    def pos(hyp_val):
	return - np.log(hyp_val)
    def neg(hyp_val):
	return - np.log(1-hyp_val)
    def cost_test(weights,training_arr,result_arr,correct_cost,correct_grad):
	(cost,cost_grad) = basic_cost_function(weights,training_arr,result_arr)
	assert cost == correct_cost
	assert (cost_grad == correct_grad).all()
    
    # actual tests

    # 1 feature - pos result
    weights	    = np.array([0.5,0.5])
    training_arr    = np.array([[1,1]])
    result_arr	    = np.array([1])
    hyp_val	    = hyp(training_arr,weights)
    
    correct_cost    = pos(hyp_val)
    correct_grad    = np.array([(hyp_val-1)*1,(hyp_val-1)*1])
    cost_test(weights,training_arr,result_arr,correct_cost,correct_grad)

    # 1 feature - neg result
    weights	    = np.array([0.5,0.5])
    training_arr    = np.array([[1,1]])
    result_arr	    = np.array([0])
    hyp_val	    = hyp(training_arr,weights)
    
    correct_cost    = neg(hyp_val) 
    correct_grad    = np.array([(hyp_val-0)*1,(hyp_val-0)*1])
    cost_test(weights,training_arr,result_arr,correct_cost,correct_grad)

    # 1 feature - pos + neg result
    weights	    = np.array([0.5,0.5])
    training_arr    = np.array([[1,0],[1,1]])
    result_arr	    = np.array([0,1])
    hyp_0	    = hyp(training_arr[0],weights)
    hyp_1	    = hyp(training_arr[1],weights)

    correct_cost_0  = neg(hyp_0)  
    correct_cost_1  = pos(hyp_1)
    correct_cost    = 0.5 * (correct_cost_0 + correct_cost_1).sum() # just 1 number
    
    correct_grad_0  = np.array([(hyp_0-0)*1,(hyp_0-0)*0])
    correct_grad_1  = np.array([(hyp_1-1)*1,(hyp_1-1)*1])
    correct_grad    = 0.5 * (correct_grad_0 + correct_grad_1)	    # 1 for each feature
    cost_test(weights,training_arr,result_arr,correct_cost,correct_grad)

    # 2 features - pos + neg result
    weights	    = np.array([0.5,0.5,0.5])
    training_arr    = np.array([[1,1,2],[1,3,4]])
    result_arr	    = np.array([1,0])
    hyp_0	    = hyp(training_arr[0],weights)
    hyp_1	    = hyp(training_arr[1],weights)

    correct_cost_0  = pos(hyp_0)  
    correct_cost_1  = neg(hyp_1)
    correct_cost    = 0.5 * (correct_cost_0 + correct_cost_1).sum() # just 1 number

    correct_grad_0  = np.array([(hyp_0-0)*1,(hyp_0-0)*1,(hyp_0-0)*2]) 
    correct_grad_1  = np.array([(hyp_1-0)*1,(hyp_1-0)*3,(hyp_1-0)*4]) 
    correct_grad    = 0.5 * (correct_grad_0 + correct_grad_1)	    # 1 for each feature

def test_sigmoid():
    """
    must handle:
    	integers
    	1x1 arrays
    	1xn arrays
    	mx1 arrays
    	mxn arrays
    probably should figure out overflow limits for float
	is it float64? how big/small can that get?
    """
    # integer
    correct = 1.0/ (1.0 + np.exp(-3) )
    assert sigmoid(3) == correct
    # 1x1
    test = np.array([3])
    correct = np.array([sigmoid(3)])
    assert (sigmoid(test) == correct).all()
    # mx1 
    test = np.array([3,4,5])
    correct = np.array([sigmoid(3),sigmoid(4),sigmoid(5)])
    assert (sigmoid(test) == correct).all()
    test = np.array([[3],[4],[5]])
    correct = np.array([[sigmoid(3)],[sigmoid(4)],[sigmoid(5)]])
    assert (sigmoid(test) == correct).all()
    
    #1xn
    test = np.array([[3,4,5]])
    correct = np.array([[sigmoid(3),sigmoid(4),sigmoid(5)]])
    assert (sigmoid(test) == correct).all()
    
    #mxn
    test = np.array([[1,2,3],
		    [4,5,6]])
    correct = np.array([[sigmoid(1),sigmoid(2),sigmoid(3)],
		    [sigmoid(4),sigmoid(5),sigmoid(6)]])
    assert (sigmoid(test) == correct).all()

def _test_grad_descent():
    """
    check that gradient descent produces the desired weights
    not yet implemented
    """
    pass

def _test_log_reg():
    """
    check that module works as intended start to finish
    not yet implemented
    """
    pass
