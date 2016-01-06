"""
This script shows a small demo of how to use log_reg to classify 1 dimensional CSV format data.
In this case we use training_example_data.csv to predict a student's final exam scores.
The dataset is fake and trivially small, and the predictions are nonsense:
This script is only for illustrative purposes

The dataset contains 10 rows of students with the following columns:
	Student ID 		int
	Test A 			int
	Test B   		int
	Test C    		int
	Test D  		int
	Final Exam 		boolean
We will be using the first four tests to predict whether a given student will pass the final exam.
"""

import numpy as np
import log_reg

PLOT_CONVERGENCE 	= True
SPREADSHEET 		= 'training_example_data.csv'

# Load CSV data
raw_data  			= np.genfromtxt(SPREADSHEET,skip_header=1,delimiter=',')
test_scores 		= raw_data[:,1:5]
exam_outcomes 		= raw_data[:,5]

# Perform the regression
# Using the weights and norm_info we can now make predictions
(weights,norm_info) = log_reg.log_reg(test_scores,exam_outcomes,
										num_examples= 10, num_features= 4,
										plot_convergence= PLOT_CONVERGENCE)

# Predict the outcomes for two new students - Barry and Tim
students = {}
students['barry'] 	= np.array([34,54,20,40])
students['tim'] 	= np.array([81,32,73,69])

for name in students:
	test_scores 	= students[name]
	exam_success 	= log_reg.hypothesis(test_scores,weights,norm_info)
	print "%s has a %.3f probability of passing the final exam" % (name,exam_success)