#!/usr/bin/python

# This Python file reads summary files located in subdirectories
# of the results directory and uses the data contained within them
# to plot a figure of false positive rate vs miss rate for a number of thresholds

import numpy as np                        # loadtxt
import matplotlib.pyplot as plt           # plot
import matplotlib.cm as cm                # rainbow
import argparse as ap                     # parser for arguments
import os                                 # linesep
import CPBUtils as ut                     # for column numbers in the summary file

# Function for calculating Cohen's Kappa statistic from a confusion matrix
def kappa_statistic(m):
	
	# Dimension
	d = m.shape[0]
	
	# Find the sums of the rows, columns and full matrix
	row_sums = m.sum(axis=1)
	col_sums = m.sum(axis=0)
	full_sum = m.sum()
	
	# Agreement is the sum along the diagonal (i.e. trace)
	po = m.trace()
	
	# Expected agreement
	pe = float(sum((row_sums[i]*col_sums[i] for i in range(d))))/full_sum
	
	return float(po - pe)/(full_sum - pe)
    

class_names = ["4C", "LVOT", "3V"]

parser = ap.ArgumentParser(description='Plot confusion matrices for the classification results')
parser.add_argument('results_directory',help="directory containing the results")
parser.add_argument('experiment_definitions',help="file containing all of the experiments to be displayed")
parser.add_argument('--output','-o',help="output summary to standard output",action="store_true")
parser.add_argument('--filter_file_name','-z',help="name of the filter file")
parser.add_argument('--neg_row','-n',help="Include a row corresponding to negative frames",action='store_true')
parser.add_argument('--summary_file_name','-s',help="name of the summary file to use within each directory",default="summary")
parser.add_argument('--use_experiment_name','-f',action="store_true",help="If this flag is set, treat the experiment_file input as the experiment name itself, rather than a file containing them")
args = parser.parse_args()

# Directory where results are stored
resultsdir = args.results_directory

if args.use_experiment_name:
	model_list = [args.experiment_definitions]
else:
	with open(args.experiment_definitions) as experiment_file :
		model_list = [line.split()[0] for line in experiment_file]

matrix_list = []
kappa_list = []

# Print an explanation message
if args.output:
	print "Class Confusion matrices:"
	print "Each matrix is a 3x4 matrix, where rows represent the true labels classes (four-chamber, left-ventricular outflow, three vessels), and each column represents the detected view (four-chamber, left-ventricular outflow, three vessels, missed)", os.linesep, os.linesep

# Loop through the lines in the experiment definition file
for modelname in model_list :

	# Determine the name of the corresponding summary file
	if args.filter_file_name is None:
		summaryname = os.path.join( resultsdir , modelname , args.summary_file_name)
	else:
		summaryname = os.path.join( resultsdir , modelname , args.filter_file_name , args.summary_file_name)


	# Load the relevant columns, ignore the first row of the confusion matrix
	start_col = (ut.sum_col_confusion_matrix_first if args.neg_row else ut.sum_col_confusion_matrix_first+4)
	finish_col = ut.sum_col_confusion_matrix_last+1
	confusion_data = np.loadtxt(summaryname,skiprows=1,usecols=range(start_col,finish_col),dtype=float)

	# Reorganise the array into a 3D stack of 2D confusion matrices
	n_rows = (4 if args.neg_row else 3)
	confmat_stack = confusion_data.reshape((-1,n_rows,4)).transpose([1,2,0])

	# Normalise within each class (such that rows of confusion matrix sum to one)
	row_sums = confmat_stack.sum(axis=1)
	# This line will often lead to runtime warnings due to videos that contain
	# no examples of a given view. This may be safely ignored because the nans
	# are dealt with via use of nanmean later
	old_settings = np.seterr()
	np.seterr(all='ignore')
	normalised_stack = confmat_stack / row_sums.reshape((n_rows,1,-1))
	np.seterr(**old_settings)


	# Average over all the non-nan videos
	confusion_matrix = np.nanmean(normalised_stack,axis=2)

	# Move the background class to the right hand side (and bottom)
	confusion_matrix = np.roll(confusion_matrix,-1,axis=1)
	if args.neg_row:
		confusion_matrix = np.roll(confusion_matrix,-1,axis=0)

	# Add this confusion matrix to the list
	matrix_list.append(confusion_matrix)
	
	# Perform an alternative normalisation for finding the kappa statistic
	# Normalise each matrix and sum so that each video has an equal contribution
	to_normalise = (confmat_stack[1:,1:,:] if args.neg_row else confmat_stack[:,1:,:])
	full_normalised_matrix = np.nansum((to_normalise/(to_normalise.sum(axis=1).sum(axis=0))), axis=2)
	kappa = kappa_statistic(full_normalised_matrix)
	kappa_list.append(kappa)

	if args.output :
		print modelname
		for row in confusion_matrix :
			print ", ".join("%.3f" % d for d in row )
		print os.linesep

for (label,cmat,kappa) in zip(model_list,matrix_list,kappa_list) :
	fig = plt.figure()
	plt.imshow(cmat,interpolation='nearest', cmap=cm.Blues, vmin=0.0,vmax=1.0)
	#plt.title(label,fontweight='bold',fontsize=30)
	if not args.output :
		print label + ", kappa = " , kappa
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=14)

	if args.neg_row:
		x_tick_marks = np.arange(len(class_names)+1)
		plt.xticks(x_tick_marks, class_names + ["Rejected/Missed"], fontsize=14)
		tick_marks = np.arange(len(class_names)+1)
		plt.yticks(tick_marks, class_names + ["None"],fontsize=14,rotation=90)
	else:
		x_tick_marks = np.arange(len(class_names)+1)
		plt.xticks(x_tick_marks, class_names + ["Missed"], fontsize=20)
		tick_marks = np.arange(len(class_names))
		plt.yticks(tick_marks, class_names,fontsize=20,rotation=90)

	plt.ylabel('True label',fontweight='bold',fontsize=25)
	plt.xlabel('Predicted label',fontweight='bold',fontsize=25)
	fig.patch.set_facecolor('white')
	plt.tight_layout()
	plt.show()
