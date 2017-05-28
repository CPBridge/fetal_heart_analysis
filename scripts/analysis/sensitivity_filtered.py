#!/usr/bin/python
import argparse as ap
from glob import glob
import os
import matplotlib.pyplot as plt
import re
import numpy as np
from CPBUtils import rocDataFromSummaryFile
import matplotlib.cm as cm                # rainbow

line_styles = ['-','--','-.',':','']
intra_observer_colour = 'orange'
inter_observer_colour = 'magenta'

parser = ap.ArgumentParser(description='Plot sensitivity plots using precaculated and pre-summarised results')
parser.add_argument('sweep_directory',help="directory containing the results of the sweep, in folders according to filter files")
parser.add_argument('--generous','-g',help="treat detections of obscured labels as correct",action="store_true")
parser.add_argument('--generous_only','-G',help="only plot the generous rate",action="store_true")
parser.add_argument('--summary_file_name','-s',default="summary",help="the root name of the summary files within each results directory")
parser.add_argument('--labels','-l',help="label the weight values along the line",action="store_true")
parser.add_argument('--key','-k',help="draw the legend",action="store_true")
parser.add_argument('--time_constants','-t',help="include only these time constants (comma-separated list, no white-space)")
parser.add_argument('--eq_fracs','-e',help="include only these equilibirum fractions (comma-separated list, no white-space)")
parser.add_argument('--inter_observer_annotation','-m',help="a summary file to use as an inter-observer annotation reference",default="none")
parser.add_argument('--intra_observer_annotation','-p',help="a summary file to use as an intra-observer annotation reference",default="none")
args = parser.parse_args()

plot_generous = args.generous or args.generous_only

# Load manual annotations if required
draw_inter_observer = args.inter_observer_annotation != "none"
if draw_inter_observer :
	if plot_generous :
		inter_observer_correct_loc_rate,inter_observer_fpr,inter_observer_fpr_generous = rocDataFromSummaryFile(args.inter_observer_annotation,True)
	else:
		inter_observer_correct_loc_rate,inter_observer_fpr = rocDataFromSummaryFile(args.inter_observer_annotation)
draw_intra_observer = args.intra_observer_annotation != "none"
if draw_intra_observer :
	if plot_generous :
		intra_observer_correct_loc_rate,intra_observer_fpr,intra_observer_fpr_generous = rocDataFromSummaryFile(args.intra_observer_annotation,True)
	else:
		intra_observer_correct_loc_rate,intra_observer_fpr = rocDataFromSummaryFile(args.intra_observer_annotation)

# Glob the names of the filters
filter_files = glob(os.path.join(args.sweep_directory , '*'))

# The first filter file name (before the __) is the filter that describes the hidden parameters
hidden_filters = [os.path.basename(f).split('__')[0] for f in filter_files]

# Parse to find the parameters
eq_fracs = [ float(re.search('_ef([0-9]+\.[0-9]+)',f).group(1)) for f in hidden_filters ]
time_constants = [ float(re.search('_tc([0-9]+\.[0-9]+)',f).group(1)) for f in hidden_filters ]
hidden_weights = [ float(re.search('_w([0-9]+\.[0-9]+)',f).group(1)) for f in hidden_filters ]

# Unique pairs of time constant and eq_frac
unique_eq_tc_pairs = list(set([(eq,tc) for eq,tc in zip(eq_fracs,time_constants)]))
unique_eq_tc_pairs.sort()

# Remove unwanted values of these parameters
if(args.eq_fracs is not None):
	chosen_fracs = [ float(x) for x in args.eq_fracs.split(',')]
	unique_eq_tc_pairs = [(eq,tc) for (eq,tc) in unique_eq_tc_pairs if (eq in chosen_fracs)]
if(args.time_constants is not None):
	chosen_tcs = [ float(x) for x in args.time_constants.split(',')]
	unique_eq_tc_pairs = [(eq,tc) for (eq,tc) in unique_eq_tc_pairs if (tc in chosen_tcs)]

# Lists of data
fpr_list = []
fpr_list_generous = []
correct_loc_rate_list = []
params_list = []
weights_list = []

# Iterate over unique pairings of equilibrium fraction and time-constant
for eq,tc in unique_eq_tc_pairs:

	weights = [w for eqeq,tctc,w in zip(eq_fracs,time_constants,hidden_weights) if ((eqeq == eq) and (tctc == tc))]
	fpr_arr = [0.0]*len(weights)
	correct_loc_rate_arr = [0.0]*len(weights)
	if plot_generous :
		fpr_arr_generous = [0.0]*len(weights)

	ind = 0
	for f,eqeq,tctc,ww in zip(filter_files,eq_fracs,time_constants,hidden_weights):
		if (eqeq == eq) and (tctc == tc):

			# Summary
			summaryname = f + '/' + args.summary_file_name
			if plot_generous :
				correct_loc_rate_arr[ind], fpr_arr[ind], fpr_arr_generous[ind] = rocDataFromSummaryFile(summaryname,True)
			else:
				correct_loc_rate_arr[ind], fpr_arr[ind] = rocDataFromSummaryFile(summaryname)

			ind += 1

	# Now need to sort the lists by their weight value, or the plots will look funny
	if(plot_generous):
		zipped = zip(weights, fpr_arr, correct_loc_rate_arr,fpr_arr_generous)
		zipped.sort() # will sort by the weight value
		weights, fpr_arr, correct_loc_rate_arr, fpr_arr_generous = zip(*zipped) # unzips
	else:
		zipped = zip(weights, fpr_arr, correct_loc_rate_arr)
		zipped.sort() # will sort by the weight value
		weights, fpr_arr, correct_loc_rate_arr = zip(*zipped) # unzips

	params_list.append(r'$\lambda^{\ast} = '+ str(eq) + r'$, $\tau = ' + str(tc) + r'$')
	fpr_list.append(fpr_arr)
	correct_loc_rate_list.append(correct_loc_rate_arr)
	weights_list.append(weights)
	if plot_generous :
		fpr_list_generous.append(fpr_arr_generous)

if len(fpr_list) >  7:
	colours = cm.gist_rainbow(np.linspace(0.0,1.0,len(fpr_list)))
else :
	colours = ['r','b','g','k','y','c','m']

# Draw the plot
fig = plt.figure(figsize=(10,6))
if not args.generous_only:
	for x,y,param_label,line_colour,w in zip(fpr_list,correct_loc_rate_list,params_list,colours,weights_list) :

		plt.plot(x,y,label=param_label,color=line_colour) # have not implemented different line styles for now as there aren't enough...

		# Add text showing the weight/threshold value, if desired
		if args.labels :
			for xx,yy,label in zip(x,y,w) :
				plt.annotate(str(label),(xx,yy),size='x-small')

	# Add in manual point if required
	if draw_inter_observer :
		plt.scatter(inter_observer_fpr,inter_observer_correct_loc_rate,label="inter-observer",color=inter_observer_colour)
	if draw_intra_observer :
		plt.scatter(intra_observer_fpr,intra_observer_correct_loc_rate,label="intra-observer",color=intra_observer_colour)

if plot_generous:
	for x,y,param_label,line_colour,w in zip(fpr_list_generous,correct_loc_rate_list,params_list,colours,weights_list) :

		plt.plot(x,y,label=param_label+ r' (generous)',color=line_colour,linestyle='--') # have not implemented different line styles for now as there aren't enough...

		# Add text showing the weight/threshold value, if desired
		if args.labels :
			for xx,yy,label in zip(x,y,w) :
				plt.annotate(str(label),(xx,yy),size='x-small')

	# Add in manual point if required
	if draw_inter_observer :
		plt.scatter(inter_observer_fpr_generous,inter_observer_correct_loc_rate,label="inter-observer (generous)",color=inter_observer_colour,marker='D')
	if draw_intra_observer :
		plt.scatter(intra_observer_fpr_generous,intra_observer_correct_loc_rate,label="intra-observer (generous)",color=intra_observer_colour,marker='D')

plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.grid(b=True, which='major', linestyle='--')
if args.key :
	plt.legend(loc=4, ncol=2,fontsize='small',scatterpoints=1)
plt.xlabel("False Positive Detection Rate",fontweight='bold')
plt.ylabel("True Positive Detection Rate",fontweight='bold')
fig.patch.set_facecolor('white')
fig.tight_layout()
plt.show()
