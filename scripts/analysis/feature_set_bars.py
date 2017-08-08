#!/usr/bin/python

import matplotlib.pyplot as plt           # plot
import argparse as ap                     # parser for arguments
import CPBUtils as ut
import numpy as np
from train_argparse import train_arg_parser

parser = ap.ArgumentParser(description='Plot bar charts comparing different rotation invariant feature sets in terms of various accuracies')

parser.add_argument('results_directory',help="directory containing the results, in folders according to model name")
parser.add_argument('train_experiment_definitions',help="file containing all of the training experiments to be displayed")
parser.add_argument('test_experiment_definitions',help="file containing all of the testing experiments to be displayed")
parser.add_argument('--type','-t',help="which forest's parameters are being varied? [class,phaseori]",default="class")
parser.add_argument('--reduction','-r',default='max',help="the operation to perform to reduce the value across the different test in order to arrive at one number, [max, min, mean, median], default 'max'")
parser.add_argument('--summary_file_name','-s',help="name of the summary file to use within each directory",default="summary")
args = parser.parse_args()

if args.reduction not in ['max','median','mean','min']:
	print 'Error, Unrecognised reduction operation ' + args.reduction + ', should be in [max,min,median,mean]'
	exit()

# Choose the reduction operation to apply to the table
if args.reduction == 'max':
	reduction_op = lambda x : np.max(x,axis=1)
if args.reduction == 'min':
	reduction_op = lambda x : np.min(x,axis=1)
if args.reduction == 'mean':
	reduction_op = lambda x : np.mean(x,axis=1)
if args.reduction == 'median':
	reduction_op = lambda x : np.median(x,axis=1)

if args.type not in ["class","ori","phase"] :
	print "Unidentified 'type' parameter, must be 'class', 'ori' or 'phase', you gave ", args.type
	exit()

# Read in the train arguments
with open(args.train_experiment_definitions,'r') as train_file:
	train_args = [line.split()[1:] for line in train_file]

# Directory where results are stored
resultsdir = args.results_directory

train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table = ut.gatherAccuracyStatsTrainTest(args.train_experiment_definitions,args.test_experiment_definitions,resultsdir,summary_file_name=args.summary_file_name,transpose=True)

# Change the classification accuracy into error for consistency
class_error_table = [ [100.0*(1.0 - y) for y in x ] for x in accuracy_table]

# Apply the reduction operation
if args.type == 'class':
	results_list = reduction_op(class_error_table)
if args.type == 'phase':
	results_list = reduction_op(phase_error_table)
if args.type == 'ori':
	results_list = reduction_op(ori_error_table)

# Get a list of the 'base' feature sets
features_list = list(set([mod.replace('_basic','').replace('_coupled','').replace('_extra','') for mod in train_model_list]))
features_list.sort()

# Check that there is one basic, coupled and extra set for each
basic_scores = []
coupled_scores = []
extra_scores = []
for feat in features_list:

	# Check the basic set
	if feat + '_basic' not in train_model_list:
		print 'There is no "basic" set for features ' + feat
		exit()
	basic_index = train_model_list.index(feat + '_basic')
	basic_feature_set_list = train_arg_parser.parse_args(train_args[basic_index]).feature_set
	if len(basic_feature_set_list) != 1 or basic_feature_set_list[0] not in ['basic','b']:
		print 'Training parameters for training model ' + feat + '_basic do not match the name'
		exit()

	# Add basic score to the list
	basic_scores.append(results_list[basic_index])

	# Check the coupled set
	if feat + '_coupled' not in train_model_list:
		print 'There is no "coupled" set for features ' + feat
		exit()
	coupled_index = train_model_list.index(feat + '_coupled')
	coupled_feature_set_list = train_arg_parser.parse_args(train_args[coupled_index]).feature_set
	if len(coupled_feature_set_list) != 1 or coupled_feature_set_list[0] not in ['couple_simple','simple','c']:
		print 'Training parameters for training model ' + feat + '_coupled do not match the name'
		exit()

	# Add coupled score to the list
	coupled_scores.append(results_list[coupled_index])

	# Check the coupled list
	if feat + '_extra' not in train_model_list:
		print 'There is no "extra" set for features ' + feat
		exit()
	extra_index = train_model_list.index(feat + '_extra')
	extra_feature_set_list = train_arg_parser.parse_args(train_args[extra_index]).feature_set
	if len(extra_feature_set_list) != 1 or extra_feature_set_list[0] not in ['couple_extra','extra','ce']:
		print 'Training parameters for training model ' + feat + '_extra do not match the name'
		exit()

	# Add the extra score to the list
	extra_scores.append(results_list[extra_index])

# Plot the results
fig = plt.figure(figsize=(10,6))
n_basic_features = len(features_list)
bar_width = 0.2
index = np.arange(n_basic_features)

bar_basic = plt.bar(index, basic_scores, bar_width,
                 color='b',
				 alpha=0.1,
                 label='basic')

bar_coupled = plt.bar(index + bar_width, coupled_scores, bar_width,
				 color='b',
				 alpha=0.6,
				 label='coupled')

bar_extra = plt.bar(index + 2*bar_width, extra_scores, bar_width,
				 color='b',
				 label='extra')

plt.xlabel('Feature Set',fontsize='large',fontweight='bold')
if args.type == 'class':
	plt.ylabel('Detection Error, (%)',fontsize='large',fontweight='bold')
if args.type == 'phase':
	plt.ylabel('Cardiac Phase Error',fontsize='large',fontweight='bold')
if args.type == 'ori':
	plt.ylabel('Orientation Error',fontsize='large',fontweight='bold')
plt.axes().yaxis.grid(b=True, which='major', linestyle='--')
#plt.title('Scores by group and gender')
plt.xticks(index + 2*bar_width, features_list, rotation=90,fontsize='small')
plt.legend(loc='lower right')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()
