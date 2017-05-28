#!/usr/bin/python

# This Python file reads summary files located in subdirectories
# of the results directory and uses the data contained within them
# to plot a figure of false positive rate vs miss rate for a number of thresholds

import numpy as np                        # loadtxt
import matplotlib.pyplot as plt           # plot
import matplotlib.cm as cm                # rainbow
from string import digits				  # digits
import argparse as ap                     # parser for arguments
import os                                 # linesep
import CPBUtils as ut                     # columns

line_styles = ['-','--','-.',':','']

intra_rater_colour = 'orange'
inter_rater_colour = 'magenta'

default_start = 0.0
default_step = 0.05
default_stop = 1.0

parser = ap.ArgumentParser(description='Plot sensitivity plots using precaculated and pre-summarised results')
parser.add_argument('results_directory',help="directory containing the results")
parser.add_argument('experiment_definitions',help="file containing all of the experiments to be displayed")
parser.add_argument('--generous','-g',help="treat detections of obscured labels as correct",action="store_true")
parser.add_argument('--key','-k',help="draw the legend",action="store_true")
parser.add_argument('--range_start','-a',type=float,help="treat detections of obscured labels as correct",default=default_start)
parser.add_argument('--range_stop','-b',type=float,help="treat detections of obscured labels as correct",default=default_stop)
parser.add_argument('--range_step','-c',type=float,help="treat detections of obscured labels as correct",default=default_step)
parser.add_argument('--decimal_precision','-n',type=int,help="file names have this many characters after decimal point",default=2)
parser.add_argument('--output','-o',help="output summary to standard output",action="store_true")
parser.add_argument('--labels','-l',help="label the weight values along the line",action="store_true")
parser.add_argument('--highlight','-H',help="highlight a certain threshold on the line with an extra marker")
parser.add_argument('--summary_file_name','-s',default="summary_",help="the root name of the summary files within each results directory")
parser.add_argument('--inter_rater_annotation','-m',help="a summary file to use as an inter-rater annotation reference",default="none")
parser.add_argument('--intra_rater_annotation','-p',help="a summary file to use as an intra-rater annotation reference",default="none")
args = parser.parse_args()

# Load manual annotations if required
draw_inter_rater = args.inter_rater_annotation != "none"
if draw_inter_rater :
	if args.generous :
		inter_rater_correct_loc_rate,inter_rater_fpr,inter_rater_fpr_generous = ut.rocDataFromSummaryFile(args.inter_rater_annotation,True)
	else:
		inter_rater_correct_loc_rate,inter_rater_fpr = ut.rocDataFromSummaryFile(args.inter_rater_annotation)
draw_intra_rater = args.intra_rater_annotation != "none"
if draw_intra_rater :
	if args.generous :
		intra_rater_correct_loc_rate,intra_rater_fpr,intra_rater_fpr_generous = ut.rocDataFromSummaryFile(args.intra_rater_annotation,True)
	else:
		intra_rater_correct_loc_rate,intra_rater_fpr = ut.rocDataFromSummaryFile(args.intra_rater_annotation)


# Directory where results are stored
resultsdir = args.results_directory
experiment_file_name = args.experiment_definitions

model_list = []
fpr_list = []
fpr_list_generous = []
correct_loc_rate_list = []

# The list of thresholds (or weight values) that form the line for each model
thresholds = np.linspace(args.range_start,args.range_stop,num=int(args.range_stop/args.range_step)+1)

# Loop through the lines in the experiment definition file
with open(experiment_file_name) as experiment_file :
	for experimentline in experiment_file :

		modelname = experimentline.split()[0]
		model_list.append(modelname.replace('square_','rec_'))

		fpr_arr = np.zeros(len(thresholds))
		correct_loc_rate_arr = np.zeros(len(thresholds))
		if args.generous :
			fpr_arr_generous = np.zeros(len(thresholds))

		for ind,t in enumerate(thresholds) :

			# Determine the name of the corresponding summary file
			summaryname = os.path.join(resultsdir , modelname , args.summary_file_name + ( ('%.' + str(args.decimal_precision) + 'f') % t) )

			if args.generous :
				correct_loc_rate_arr[ind], fpr_arr[ind], fpr_arr_generous[ind] = ut.rocDataFromSummaryFile(summaryname,True)
			else:
				correct_loc_rate_arr[ind], fpr_arr[ind] = ut.rocDataFromSummaryFile(summaryname)

		fpr_list.append(fpr_arr)
		correct_loc_rate_list.append(correct_loc_rate_arr)
		if args.generous :
			fpr_list_generous.append(fpr_arr_generous)

#colours = cm.jet(np.linspace(0.0,1.0,len(model_list)))
if len(model_list) >  7:
	colours = cm.gist_rainbow(np.linspace(0.0,1.0,len(model_list)))
else :
	colours = ['r','b','g','k','y','c','m']

# Get a unique marker type for each image representation
reps_list = [model.translate(None,digits) for model in model_list] # remove JKM numbers to leave just the image representations
unique_reps_list = list(set(reps_list))                            # remove duplicates
rep_styles = [line_styles[unique_reps_list.index(rep)] for rep in reps_list]   # assign a marker to each model according to its representation


# Draw the plot
fig = plt.figure(figsize=(10,6))
lines_list = []
for x,y,model_label,line_colour,ls in zip(fpr_list,correct_loc_rate_list,model_list,colours,rep_styles) :

	plt.plot(x,y,label=model_label,color=line_colour) # have not implemented different line styles for now as there aren't enough...

	# Add text showing the weight/threshold value, if desired
	if args.labels :
		for xx,yy,label in zip(x,y,thresholds) :
			plt.annotate(label,(xx,yy),size='x-small')

# Add in manual point if required
if draw_inter_rater :
	plt.scatter(inter_rater_fpr,inter_rater_correct_loc_rate,label="inter-rater",color=inter_rater_colour)
if draw_intra_rater :
	plt.scatter(intra_rater_fpr,intra_rater_correct_loc_rate,label="intra-rater",color=intra_rater_colour)

# Work out which point is to be highlighted (if required) and plot it
if args.highlight is not None:
	thresh_strings = [( ('%.' + str(args.decimal_precision) + 'f') % t) for t in thresholds]
	if args.highlight not in thresh_strings:
		print "ERROR: Could not find a threshold matching", args.highlight, "requst will be ignored"
	else :
		# Get the index of this threshold/weight
		hind = thresh_strings.index(args.highlight)
		# Plot an extra point for each of the models
		for x,y,line_colour in zip(fpr_list,correct_loc_rate_list,colours) :
			plt.scatter(x[hind],y[hind],color=line_colour,marker='x',s=160)

# Draw the generous line if required
if args.generous :
	for x,y,model_label,line_colour,ls in zip(fpr_list_generous,correct_loc_rate_list,model_list,colours,rep_styles) :
		plt.plot(x,y,label=model_label+" (generous)",color=line_colour,linestyle='--') # have not implemented different line styles for now as there aren't enough...
		# Add text showing the weight/threshold value, if desired
		if args.labels :
			for xx,yy,label in zip(x,y,thresholds) :
				plt.annotate(label,(xx,yy),size='x-small')

	# Add in manual point if required
	if draw_inter_rater :
		plt.scatter(inter_rater_fpr_generous,inter_rater_correct_loc_rate,label="inter-rater (generous)",color=inter_rater_colour,marker='D')
	if draw_intra_rater :
		plt.scatter(intra_rater_fpr_generous,intra_rater_correct_loc_rate,label="intra-rater (generous)",color=intra_rater_colour,marker='D')

plt.xlabel("False Positive Rate",fontweight='bold',fontsize='large')
plt.xlim(0.0,1.0)
plt.ylabel("True Positive Rate",fontweight='bold',fontsize='large')
plt.ylim(0.0,1.0)
# plt.plot([0.0,1.0],[0.0,1.0],linestyle='--',color=[0.5,0.5,0.5]) # draw a y=x line
fig.patch.set_facecolor('white')

plt.grid(b=True, which='major', linestyle='--')
if args.key :
	plt.legend(loc=4, ncol=2,fontsize='small',scatterpoints=1)
fig.tight_layout()
plt.show()

# Print to standard outputif required
if args.output :
	print "False Positive Detection Rate at multiple thresholds"
	print "threshold," , ", ".join("%.3f" % x for x in thresholds )
	for x,model_label in zip(fpr_list,model_list):
		print model_label + ",", ", ".join("%.3f" % d for d in x )

	print os.linesep

	if args.generous :
		print "False Positive Detection Rate at multiple thresholds using 'generous' criteria"
		print "threshold," , ", ".join("%.3f" % x for x in thresholds )
		for x,model_label in zip(fpr_list_generous,model_list):
			print model_label + ",", ", ".join("%.3f" % d for d in x )

	print os.linesep

	print "True Positive Detection Rate at multiple thresholds"
	print "threshold," , ", ".join("%.3f" % x for x in thresholds )
	for x,model_label in zip(correct_loc_rate_list,model_list):
		print model_label + ",", ", ".join("%.3f" % d for d in x )
