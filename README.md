# Logistic Regression Module

#### Overview

This module fits a logistic regression to a set of training data. The results of the regression can then be used to classify new data. For example, this module could be used to predict if a student will pass or fail an exam, given previous data on school performance.

Here is a 1-D logistic function fit to some arbitrary data:

![examplelogreg](https://cloud.githubusercontent.com/assets/12790824/10595595/d0a908c2-7725-11e5-9af6-741161630aaa.GIF)

In this case, the training data could represent a students score on a previous test, and the result could be the probability of the student passing a future test.

#### Brief Run-through

To perform a regression on a data set, you need an array of the training data, and an array of the result data.
This module performs the following functions:
* format_array	- format the training array so it has the correct dimensions
* add_bias	- adds the necessary bias term to formatted array
* normalize	- get the normalization info for the training array (optional)
* apply_norm	- apply the normalization to the training array (optional)
* log_reg	- calculate the weights (output) by performing logistic regression

Once the regression has been performed, the weights can be used to classify new data.
To classify new data, use the hypothesis function:
* this function takes a single data point and normalization info
* returns likelihood as a value from 0 to 1

#### Testing:
Unit tests for use with py.test are contained within test_log_reg.py
Although it is functional AFAIK, this module is a work in progress and more tests need to be written. 

#### Moldule Dependencies:
* numpy
* matplotlib

#### Future Work:
* add an example case to demonstrate use of module
* implement test_log_reg
* implement test_grad_descent
* implement cost functions with regularization
* continue work on visualization tools

#### Potential Future Work:
* implement non linear hypotheses
