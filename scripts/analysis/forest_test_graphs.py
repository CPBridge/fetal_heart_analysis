#!/usr/bin/python

import matplotlib.pyplot as plt           # plot
import matplotlib.cm as cm                # rainbow
import argparse as ap                     # parser for arguments
import CPBUtils as ut
from test_argparse import test_arg_parser
import numpy as np

parser = ap.ArgumentParser(description='Plot accuracy against time plots for classification/orientation/phase for a number of different forest configurations')
parser.add_argument('results_directory',help="directory containing the results, in folders according to model name")
parser.add_argument('train_experiment_definitions',help="file containing all of the training experiments to be displayed")
parser.add_argument('test_experiment_definitions',help="file containing all of the testing experiments to be displayed")
parser.add_argument('--type','-t',help="which forest's parameters are being varied? [class,phaseori]",default="class")
parser.add_argument('--output','-o',help="output summary to standard output",action="store_true")
parser.add_argument('--legend','-l',help="display the legend",action="store_true")
parser.add_argument('--annotations','-a',help="annotate the levels next to each point, may be very cluttered",action="store_true")
parser.add_argument('--summary_file_name','-s',help="name of the summary file to use within each directory",default="summary")
parser.add_argument('--no_display','-n',action='store_true',help='Supress displaying the plot')
parser.add_argument('--filter_file_name','-z',help='Results are found in a directory relating to the filterfile used with this name')
parser.add_argument('--write_pdf_stem','-w',help='Write a pdf of each plot with this stem')
parser.add_argument('--xlimits_detection','-x',help='min and max x-axis values for detection error plot, separated by a comma with no space (e.g. xmin,xmax)')
parser.add_argument('--xlimits_ori','-v',help='min and max x-axis values for orientation error plot, separated by a comma with no space (e.g. xmin,xmax)')
parser.add_argument('--xlimits_phase','-u',help='min and max x-axis values for phase error plot, separated by a comma with no space (e.g. xmin,xmax)')
parser.add_argument('--ylimits','-y',help='min and max y-axis (time) values for all plots, separated by a comma with no space (e.g. y_min,y_max)')
parser.add_argument('--legend_location','-L',help="location of the legend (use the relevant matplotlib specifier)",default="upper right")
args = parser.parse_args()

if args.ylimits is not None:
	y_min = int(args.ylimits.split(',')[0])
	y_max = int(args.ylimits.split(',')[1])

if args.xlimits_detection is not None:
	x_min_class = int(args.xlimits_detection.split(',')[0])
	x_max_class = int(args.xlimits_detection.split(',')[1])

if args.xlimits_ori is not None:
	x_min_ori = float(args.xlimits_ori.split(',')[0])
	x_max_ori = float(args.xlimits_ori.split(',')[1])

if args.xlimits_phase is not None:
	x_min_phase = float(args.xlimits_phase.split(',')[0])
	x_max_phase = float(args.xlimits_phase.split(',')[1])

annotation_x_offset_class = 2
annotation_x_offset_angle = 0.02
annotation_y_offset = 0

# Directory where results are stored
resultsdir = args.results_directory

if args.type not in ["class","phaseori"] :
	print "Unidentified 'type' parameter, must be 'class' or 'phaseori', you gave ", args.type
	exit()

if args.filter_file_name is None:
	train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table = ut.gatherAccuracyStatsTrainTest(args.train_experiment_definitions,args.test_experiment_definitions,resultsdir,summary_file_name=args.summary_file_name,transpose=True)
else:
	train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table = ut.gatherAccuracyStatsTrainTest(args.train_experiment_definitions,args.test_experiment_definitions,resultsdir,summary_file_name=args.summary_file_name,transpose=True,filtername=args.filter_file_name)

# Read in the test arguments
with open(args.test_experiment_definitions,'r') as test_file:
	test_args = [line.split()[1:] for line in test_file]

# Get a list of the tree numbers in each model
if args.type == 'class':
	treenums_all_list = [test_arg_parser.parse_args(a).n_trees for a in test_args]
	levels_all_list = [test_arg_parser.parse_args(a).n_tree_levels for a in test_args]
elif args.type == 'phaseori':
	treenums_all_list = [test_arg_parser.parse_args(a).n_trees_phase for a in test_args]
	levels_all_list = [test_arg_parser.parse_args(a).n_tree_levels_phase for a in test_args]
treenums_unique_list = sorted(list(set(treenums_all_list)))

# Work out a colour map
colourindex = np.linspace(0.0,1.0,num=len(treenums_unique_list))
colours = cm.jet(colourindex)

# Loop over differnt training models
for train_model, time_list, accuracy_list, ori_error_list, phase_error_list in zip(train_model_list, time_table, accuracy_table, ori_error_table, phase_error_table) :

	# Classification Error
	# -------------------

	fig = plt.figure(figsize=(10,6))
	# Loop over tree numbers
	for treenum,col in zip(treenums_unique_list,colours) :
		relevant_indices = [i for i in range(len(test_model_list)) if treenums_all_list[i] == treenum]
		relevant_indices.sort(key=lambda x: levels_all_list[x])
		relevant_time_list = [time_list[i] for i in relevant_indices]
		relevant_accuracy_list = [ 100.0*(1.0-accuracy_list[i]) for i in relevant_indices]

		plt.plot(relevant_accuracy_list,relevant_time_list,'-o',color=col,label=str(treenum)+' trees')

		if args.annotations :
			relevant_levels_list = [levels_all_list[i] for i in relevant_indices]
			for x,y,level in zip(relevant_accuracy_list,relevant_time_list,relevant_levels_list) :
				plt.annotate(str(level),xy=(x,y),xytext=(5, 5),
					textcoords='offset points', ha='right', va='bottom',fontsize='small')

	plt.xlabel("Detection Error (%)",fontweight='bold', fontsize='large')
	plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.legend(loc=args.legend_location)
	if args.xlimits_detection is not None:
		plt.xlim(x_min_class,x_max_class)
	if args.ylimits is not None:
		plt.ylim(y_min,y_max)
	plt.grid(b=True, which='major', linestyle='--')
	fig.patch.set_facecolor('white')
	fig.tight_layout()
	if args.write_pdf_stem is not None:
		plt.savefig(args.write_pdf_stem + '_class.pdf',bbox_inches='tight')
	print train_model

	# Orientation Error
	# -------------------

	fig = plt.figure(figsize=(10,6))
	# Loop over tree numbers
	for treenum,col in zip(treenums_unique_list,colours) :
		relevant_indices = [i for i in range(len(test_model_list)) if treenums_all_list[i] == treenum]
		relevant_indices.sort(key=lambda x: levels_all_list[x])
		relevant_time_list = [time_list[i] for i in relevant_indices]
		relevant_accuracy_list = [ ori_error_list[i] for i in relevant_indices]

		plt.plot(relevant_accuracy_list,relevant_time_list,'-o',color=col,label=str(treenum)+' trees')

		if args.annotations :
			relevant_levels_list = [levels_all_list[i] for i in relevant_indices]
			for x,y,level in zip(relevant_accuracy_list,relevant_time_list,relevant_levels_list) :
				plt.annotate(str(level),xy=(x,y),xytext=(5, 5),
					textcoords='offset points', ha='right', va='bottom',fontsize='small')


	plt.xlabel("Normalised Orientation Error",fontweight='bold', fontsize='large')
	plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.legend(loc=args.legend_location)
	if args.xlimits_ori is not None:
		plt.xlim(x_min_ori,x_max_ori)
	if args.ylimits is not None:
		plt.ylim(y_min,y_max)
	plt.grid(b=True, which='major', linestyle='--')
	fig.patch.set_facecolor('white')
	fig.tight_layout()
	if args.write_pdf_stem is not None:
		plt.savefig(args.write_pdf_stem + '_ori.pdf',bbox_inches='tight')

	# Cardiac Phase Error:
	# -------------------

	fig = plt.figure(figsize=(10,6))
	# Loop over tree numbers
	for treenum,col in zip(treenums_unique_list,colours) :
		relevant_indices = [i for i in range(len(test_model_list)) if treenums_all_list[i] == treenum]
		relevant_indices.sort(key=lambda x: levels_all_list[x])
		relevant_time_list = [time_list[i] for i in relevant_indices]
		relevant_accuracy_list = [ phase_error_list[i] for i in relevant_indices]

		plt.plot(relevant_accuracy_list,relevant_time_list,'-o',color=col,label=str(treenum)+' trees')

		if args.annotations :
			relevant_levels_list = [levels_all_list[i] for i in relevant_indices]
			for x,y,level in zip(relevant_accuracy_list,relevant_time_list,relevant_levels_list) :
				plt.annotate(str(level),xy=(x,y),xytext=(5, 5),
					textcoords='offset points', ha='right', va='bottom',fontsize='small')

	plt.xlabel("Normalised Cardiac Phase Error",fontweight='bold', fontsize='large')
	plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.legend(loc=args.legend_location)
	if args.xlimits_phase is not None:
		plt.xlim(x_min_phase,x_max_phase)
	if args.ylimits is not None:
		plt.ylim(y_min,y_max)
	plt.grid(b=True, which='major', linestyle='--')
	fig.patch.set_facecolor('white')
	fig.tight_layout()
	if args.write_pdf_stem is not None:
		plt.savefig(args.write_pdf_stem + '_phase.pdf',bbox_inches='tight')

	if not args.no_display :
		plt.show()
