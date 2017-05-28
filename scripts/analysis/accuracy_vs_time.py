#!/usr/bin/python

# This Python file reads summary files located in subdirectories
# of the results directory and uses the data contained within them
# to plot a figure of accuracy vs speed for different models

import numpy as np                        # loadtxt
import matplotlib.pyplot as plt           # plot
import matplotlib.cm as cm                # rainbow
import argparse as ap                     # parser for arguments
from string import digits				  # digits
import math                               # ceil
import CPBUtils as ut                     # summary file columns

# Plot area
x_lim_upper_class = 90
x_lim_upper_angle = 0.5
x_lim_upper_phase = 0.5
y_lim_low = 0
y_lim_upper = 40

# A list of scatter markers that can be used (the first are used first)
scatter_markers = "Do^spx*8<>1234"
intra_observer_colour = 'orange'
inter_observer_colour = 'magenta'
intra_observer_linestyle = inter_observer_linestyle = '--'
intra_observer_linewidth = inter_observer_linewidth = 3

parser = ap.ArgumentParser(description='Plot accuracy against time plots for classification/orientation/phase')
parser.add_argument('results_directory',help="directory containing the results, in folders according to model name")
parser.add_argument('test_experiment_definitions',help="file containing all of the testing experiments to be displayed")
parser.add_argument('train_experiment_definitions',help="file containing all of the training experiments to be displayed")
parser.add_argument('--output','-o',help="output summary to standard output",action="store_true")
parser.add_argument('--legend','-l',help="display the legend",action="store_true")
parser.add_argument('--annotations','-a',help="annotate the experiment names next to each point, may be very cluttered",action="store_true")
parser.add_argument('--summary_file_name','-s',help="name of the summary file to use within each directory",default="summary")
parser.add_argument('--inter_observer_annotation','-m',help="a summary file to use as an inter-observer annotation reference",default="none")
parser.add_argument('--intra_observer_annotation','-n',help="a summary file to use as an intra-observer annotation reference",default="none")
args = parser.parse_args()

# Directory where results are stored
resultsdir = args.results_directory

train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table = ut.gatherAccuracyStatsTrainTest(args.train_experiment_definitions,args.test_experiment_definitions,resultsdir,summary_file_name=args.summary_file_name)

if len(train_model_list) > 1 and len(test_model_list) > 1 :
	print "ERROR: No way to cope with multiple train and test files"
	exit()

# Choose which list is being varied
if (len(test_model_list) > 1) :
	varied_model_list = test_model_list
	accuracy_list = [ x[0] for x in accuracy_table]
	ori_error_list = [ x[0] for x in ori_error_table]
	phase_error_list = [ x[0] for x in phase_error_table]
	time_list = [ x[0] for x in time_table]
else :
	varied_model_list = train_model_list
	accuracy_list = accuracy_table[0]
	ori_error_list = ori_error_table[0]
	phase_error_list = phase_error_table[0]
	time_list = time_table[0]

# Load manual annotations if required
draw_inter_observer = args.inter_observer_annotation != "none"
if draw_inter_rater :
	inter_observer_accuracy,_,inter_observer_ori_error,inter_observer_phase_error = ut.accuracyFromSummaryFile(args.inter_observer_annotation)

# Load manual annotations if required
draw_intra_observer = args.intra_observer_annotation != "none"
if draw_intra_observer :
	intra_observer_accuracy,_,intra_observer_ori_error,intra_observer_phase_error = ut.accuracyFromSummaryFile(args.intra_observer_annotation)

# Get a unique marker type for each image representation
reps_list = [model.translate(None,digits) for model in varied_model_list] # remove JKM numbers to leave just the image representations
unique_reps_list = list(set(reps_list))                            # remove duplicates
rep_markers = [scatter_markers[unique_reps_list.index(rep)] for rep in reps_list]   # assign a marker to each model according to its representation

# Dumb loop to assign colours to each representation in order according to the order in which are
# listed in the experiment definition file
colourindex = np.zeros(len(reps_list),dtype=float)
for outerrepstring in unique_reps_list :
	count = 0
	total = np.count_nonzero([innerrepstring == outerrepstring for innerrepstring in reps_list])

	for list_pos,innerrepstring in enumerate(reps_list) :
		if innerrepstring == outerrepstring :
			colourindex[list_pos] = float(count + 1) / float(total + 1)
			count += 1

colours = cm.jet(colourindex)


# Draw the classification accuracy vs speed plot
classification_error_list = [100.0*(1.0-x) for x in accuracy_list]
fig = plt.figure(figsize=(10,6))
# Plot once with just a transparent face
for (y,x,col,lab,mark) in zip(classification_error_list,time_list,colours,varied_model_list,rep_markers)  :
	plt.scatter(y,x,edgecolors='k',facecolors=col,marker=mark,label=lab.replace('square_','rec_'),s=160,alpha=0.5)
# Plot again with just a black outline - this will then be 'above' all the colour faces
for (y,x,col,lab,mark) in zip(classification_error_list,time_list,colours,varied_model_list,rep_markers)  :
	plt.scatter(y,x,edgecolors='k',facecolors='none',marker=mark,label=None,s=160)
plt.xlabel("Detection Error (%)",fontweight='bold', fontsize='large')
plt.xlim(0,x_lim_upper_class)
plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
plt.ylim(0,y_lim_upper)
plt.tick_params(axis='both', which='major', labelsize=16)
fig.patch.set_facecolor('white')

# Add written labels to name each experiment if desired
if args.annotations :
	for x,y,label in zip(classification_error_list,time_list,varied_model_list) :
		plt.annotate(label,(x,y),size='x-small')

# Add manual accuracy lines
if draw_inter_observer :
	plt.vlines(100.0*(1.0-inter_observer_accuracy),0,50,linestyle=inter_observer_linestyle,colors=inter_observer_colour,lw=inter_observer_linewidth)
if draw_intra_observer :
	plt.vlines(100.0*(1.0-intra_observer_accuracy),0,50,linestyle=intra_observer_linestyle,colors=intra_observer_colour,lw=intra_observer_linewidth)

plt.grid(b=True, which='major', linestyle='--')
fig.tight_layout()

# Show the legend (will be too big and need to be manually removed from the SVG with Inkscape!)
if args.legend :
	plt.legend(ncol=2*len(varied_model_list),scatterpoints=1)
plt.show()

# Draw the orientation error vs speed plot
fig = plt.figure(figsize=(10,6))
# Plot once with just a transparent face
for (y,x,col,lab,mark) in zip(ori_error_list,time_list,colours,varied_model_list,rep_markers)  :
	plt.scatter(y,x,edgecolors='k',facecolors=col,marker=mark,label=lab.replace('square_','rec_'),s=160,alpha=0.5)
# Plot again with just a black outline - this will then be 'above' all the colour faces
for (y,x,col,lab,mark) in zip(ori_error_list,time_list,colours,varied_model_list,rep_markers)  :
	plt.scatter(y,x,edgecolors='k',facecolors="none",marker=mark,label=None,s=160)
plt.xlabel("Normalised Orientation Error",fontweight='bold', fontsize='large')
plt.xlim(0.0,x_lim_upper_angle)
plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
plt.ylim(0,y_lim_upper)
plt.tick_params(axis='both', which='major', labelsize=16)
fig.patch.set_facecolor('white')

# Add written labels to name each experiment if desired
if args.annotations :
	for x,y,label in zip(ori_error_list,time_list,varied_model_list) :
		plt.annotate(label,(x,y),size='x-small')

plt.grid(b=True, which='major', linestyle='--')

# Add manual accuracy lines
if draw_inter_observer :
	plt.vlines(inter_observer_ori_error,0,50,linestyle=inter_observer_linestyle,colors=inter_observer_colour,lw=inter_observer_linewidth)
if draw_intra_observer :
	plt.vlines(intra_observer_ori_error,0,50,linestyle=intra_observer_linestyle,colors=intra_observer_colour,lw=intra_observer_linewidth)

fig.tight_layout()
if args.legend :
	plt.legend(ncol=2*len(varied_model_list),scatterpoints=1)
plt.show()

# Draw the phase error vs speed plot
fig = plt.figure(figsize=(10,6))
# Plot once with just a transparent face
for (y,x,col,lab,mark) in zip(phase_error_list,time_list,colours,varied_model_list,rep_markers)  :
	plt.scatter(y,x,edgecolors='k',facecolors=col,marker=mark,label=lab.replace('square_','rec_'),s=160,alpha=0.5)
# Plot again with just a black outline - this will then be 'above' all the colour faces
for (y,x,col,lab,mark) in zip(phase_error_list,time_list,colours,varied_model_list,rep_markers)  :
	plt.scatter(y,x,edgecolors='k',facecolors='none',marker=mark,label=None,s=160)
plt.xlabel("Normalised Cardiac Phase Error",fontweight='bold', fontsize='large')
plt.xlim(0.0,x_lim_upper_phase)
plt.ylabel("Average Time per Frame (ms)",fontweight='bold', fontsize='large')
plt.ylim(0,y_lim_upper)
plt.tick_params(axis='both', which='major', labelsize=16)
fig.patch.set_facecolor('white')

# Add written labels to name each experiment if desired
if args.annotations :
	for x,y,label in zip(phase_error_list,time_list,varied_model_list) :
		plt.annotate(label,(x,y),size='x-small')

# Add a manual accuracy line
if draw_inter_observer :
	plt.vlines(inter_observer_phase_error,0,50,linestyle=inter_observer_linestyle,colors=inter_observer_colour,lw=inter_observer_linewidth)
if draw_intra_observer :
	plt.vlines(intra_observer_phase_error,0,50,linestyle=intra_observer_linestyle,colors=intra_observer_colour,lw=intra_observer_linewidth)

plt.grid(b=True, which='major', linestyle='--')

fig.tight_layout()
if args.legend :
	plt.legend(ncol=2*len(varied_model_list),scatterpoints=1)
plt.show()

# Output to standard output if requested
if args.output :
	print "model, ", ", ".join(varied_model_list)
	print "time_per_frame, ", ", ".join("%.3f" % x for x in time_list)
	print "classification_error, ", ", ".join("%.3f" % x for x in classification_error_list)
	print "normalised_orientation_error, ", ", ".join("%.3f" % x for x in ori_error_list)
	print "normalised_cardiac_phase_error, ", ", ".join("%.3f" % x for x in phase_error_list)
