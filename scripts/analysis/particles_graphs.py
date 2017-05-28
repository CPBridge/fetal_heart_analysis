#!/usr/bin/python

import matplotlib.pyplot as plt           # plot
import matplotlib.cm as cm
import argparse as ap                     # parser for arguments
import CPBUtils as ut
from test_argparse import test_arg_parser
import numpy as np

intra_observer_colour = 'orange'
inter_observer_colour = 'magenta'
intra_observer_linestyle = inter_observer_linestyle = '--'
intra_observer_linewidth = inter_observer_linewidth = 3

parser = ap.ArgumentParser(description='Plot accuracy against time plots for classification/orientation/phase for different numbers of particles')
parser.add_argument('results_directory',help="directory containing the results, in folders according to model name")
parser.add_argument('--train_experiment_definitions','-r',nargs='+',help="file containing all of the training experiments to be displayed")
parser.add_argument('--test_experiment_definitions','-e',nargs='+',help="file containing all of the testing experiments to be displayed")
parser.add_argument('--filter_file_names','-z',nargs='+',help='results are found in a directory relating to the filterfile used with this name')
parser.add_argument('--output','-o',help="output summary to standard output",action="store_true")
parser.add_argument('--legend','-l',help="display the legend",action="store_true")
parser.add_argument('--annotations','-a',help="annotate the levels next to each point, may be very cluttered",action="store_true")
parser.add_argument('--summary_file_name','-s',help="name of the summary file to use within each directory",default="summary")
parser.add_argument('--inter_observer_annotation','-m',help="a summary file to use as an inter-rater annotation reference",default="none")
parser.add_argument('--intra_observer_annotation','-n',help="a summary file to use as an intra-rater annotation reference",default="none")
parser.add_argument('--structures_results','-S',help="also plot structures results",action="store_true")
parser.add_argument('--time_limits','-t',help="limits on the time axis for structure plot only",type=int,nargs=2)
parser.add_argument('--localisation_limits','-L',help="limits on the localisation axis",type=float,nargs=2)
parser.add_argument('--legend_entries','-k',help="how the models should appear in the legend",nargs='+')
#parser.add_argument('--no_display','-n',action='store_true',help='Supress displaying the plot')
#parser.add_argument('--write_pdf_stem','-w',help='Write a pdf of each plot with this stem')
args = parser.parse_args()

# Directory where results are stored
resultsdir = args.results_directory

if (len(args.train_experiment_definitions) != len(args.test_experiment_definitions)) or (len(args.train_experiment_definitions) != len(args.filter_file_names)):
	print "ERROR: Number of arguments for train_experiment_definitions, test_experiment_definitions and filter_file_names must match"
	exit()

# Load manual annotations if required
draw_inter_observer = args.inter_observer_annotation != "none"
if draw_inter_observer :
	inter_observer_accuracy,_,inter_observer_ori_error,inter_observer_phase_error = ut.accuracyFromSummaryFile(args.inter_observer_annotation)

# Load manual annotations if required
draw_intra_observer = args.intra_observer_annotation != "none"
if draw_intra_observer :
	intra_observer_accuracy,_,intra_observer_ori_error,intra_observer_phase_error = ut.accuracyFromSummaryFile(args.intra_observer_annotation)

# Detection Figure
detection_fig = plt.figure(figsize=(10,6))
plt.xlabel("Detection Error (%)",fontweight='bold', fontsize='large')
plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(b=True, which='major', linestyle='--')
detection_fig.patch.set_facecolor('white')
detection_fig.tight_layout()
# Add manual accuracy lines
if draw_inter_observer :
	plt.axvline(100.0*(1.0-inter_observer_accuracy),linestyle=inter_observer_linestyle,color=inter_observer_colour,lw=inter_observer_linewidth)
if draw_intra_observer :
	plt.axvline(100.0*(1.0-intra_observer_accuracy),linestyle=intra_observer_linestyle,color=intra_observer_colour,lw=intra_observer_linewidth)

# Orientation Figure
ori_fig = plt.figure(figsize=(10,6))
plt.xlabel("Normalised Orientation Error",fontweight='bold', fontsize='large')
plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(b=True, which='major', linestyle='--')
# Add manual accuracy lines
if draw_inter_observer :
	plt.axvline(inter_observer_ori_error,linestyle=inter_observer_linestyle,color=inter_observer_colour,lw=inter_observer_linewidth)
if draw_intra_observer :
	plt.axvline(intra_observer_ori_error,linestyle=intra_observer_linestyle,color=intra_observer_colour,lw=intra_observer_linewidth)
ori_fig.patch.set_facecolor('white')
ori_fig.tight_layout()

# Phase figure
phase_fig = plt.figure(figsize=(10,6))
plt.xlabel("Normalised Cardiac Phase Error",fontweight='bold', fontsize='large')
plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(b=True, which='major', linestyle='--')
# Add a manual accuracy line
if draw_inter_observer :
	plt.axvline(inter_observer_phase_error,linestyle=inter_observer_linestyle,color=inter_observer_colour,lw=inter_observer_linewidth)
if draw_intra_observer :
	plt.axvline(intra_observer_phase_error,linestyle=intra_observer_linestyle,color=intra_observer_colour,lw=intra_observer_linewidth)
phase_fig.patch.set_facecolor('white')
phase_fig.tight_layout()

# Structures figure
if args.structures_results:
	structs_fig = plt.figure(figsize=(10,6))
	plt.xlabel("Localisation Error",fontweight='bold', fontsize='large')
	plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.grid(b=True, which='major', linestyle='--')
	structs_fig.patch.set_facecolor('white')
	structs_fig.tight_layout()

# Loop over the train and test files
for train_model_file, test_model_file, filter_name in zip(args.train_experiment_definitions,args.test_experiment_definitions,args.filter_file_names):

	train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table = ut.gatherAccuracyStatsTrainTest(train_model_file,test_model_file,resultsdir,summary_file_name=args.summary_file_name,transpose=True,filtername=filter_name)

	# Can only process training experiment files with one training experiments
	if len(train_model_list) > 1:
		print "ERROR: Training experiment file", train_model_file , "contains multiple experiments"
		exit()

	# Get the lists for the first and only training experiment
	train_model = train_model_list[0].replace('square_','rec_')
	legend_entry = train_model if args.legend_entries is None else args.legend_entries[args.test_experiment_definitions.index(test_model_file)]
	time_list = time_table[0]
	accuracy_list = accuracy_table[0]
	ori_error_list = ori_error_table[0]
	phase_error_list = phase_error_table[0]

	# Read in structures results
	if args.structures_results:
		struct_dist_table,_,_ = ut.getStructureDataTrainTest(train_model_file,test_model_file,filter_name,resultsdir,summary_file_name=args.summary_file_name,transpose=True)
		struct_dist_list = struct_dist_table[0]

	# Read in the test arguments
	with open(test_model_file,'r') as test_file:
		test_args = [line.split()[1:] for line in test_file]

	# Get a list of the tree numbers in each model
	particles_list = [test_arg_parser.parse_args(a).n_particles for a in test_args]

	# Detection Error
	plt.figure(detection_fig.number)
	detection_error = [ 100.0*(1.0-x) for x in accuracy_list]
	plt.plot(detection_error,time_list,'-o',label=legend_entry)
	for x,y,p in zip(detection_error,time_list,particles_list):
		plt.annotate(str(p),
		xy=(x, y), xytext=(40, 20),
		textcoords='offset points', ha='right', va='bottom',fontsize='small',
		arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	# Orientation Error
	plt.figure(ori_fig.number)
	plt.plot(ori_error_list,time_list,'-o',label=legend_entry)
	for x,y,p in zip(ori_error_list,time_list,particles_list):
		plt.annotate(str(p),
		xy=(x, y), xytext=(40, 20),
		textcoords='offset points', ha='right', va='bottom',fontsize='small',
		arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	# Cardiac Phase Error
	plt.figure(phase_fig.number)
	plt.plot(phase_error_list,time_list,'-o',label=legend_entry)
	for x,y,p in zip(phase_error_list,time_list,particles_list):
		plt.annotate(str(p),
		xy=(x, y), xytext=(40, 20),
		textcoords='offset points', ha='right', va='bottom',fontsize='small',
		arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	# Structure Localisation
	if args.structures_results:
		plt.figure(structs_fig.number)

		# Choose which colours to use for each structure
		colourindex = np.linspace(0.0,1.0,num=len(struct_dist_list[0]))
		colours = cm.rainbow(colourindex)

		for struct,c in zip(struct_dist_list[0],colours) :
			this_struct_dist_list = [dist_dict[struct] for dist_dict in struct_dist_list]
			plt.plot(this_struct_dist_list,time_list,'-o',label=struct,color=c)

		# Choose the structure with the lowest localisation error as the location
		# for the
		annotate_x_points = [d[min(d,key=d.get)] for d in struct_dist_list]

		for x,y,p in zip(annotate_x_points,time_list,particles_list):
			plt.annotate(str(p),
			xy=(x, y), xytext=(-40, 0),
			textcoords='offset points', ha='right', va='center',fontsize='small',
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

		if args.time_limits is not None:
			plt.ylim(args.time_limits)
		if args.localisation_limits is not None:
			plt.xlim(args.localisation_limits)
		plt.legend(fontsize='small')


# Add legends
for num in [detection_fig.number,ori_fig.number,phase_fig.number]:
	plt.figure(num)
	plt.legend()

plt.show()
