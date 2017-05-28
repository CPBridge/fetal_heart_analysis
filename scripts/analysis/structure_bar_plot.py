#!/usr/bin/python

import matplotlib.pyplot as plt           # plot
import argparse as ap                     # parser for arguments
import CPBUtils as ut

parser = ap.ArgumentParser(description='Plot a bar graph of the errors for the different structures in a summary file')
parser.add_argument('summary_file',help="the summary file")
parser.add_argument('--ylimit','-y',type=float,help="The upper y axis limit")
args = parser.parse_args()

distance_error,_,_ = ut.getStructureDataFromSummaryFile(args.summary_file)

fig = plt.figure(figsize=(10,6))

index = range(len(distance_error))
error_list = [v for k,v in distance_error.iteritems()]
structure_list = [k for k in distance_error]
bar_width = 0.5
midpoints = [x+bar_width/2.0 for x in index]

plt.bar(index, error_list, bar_width,color='cyan')
plt.xticks(midpoints, structure_list, rotation=90,fontsize='small',fontweight='bold')
plt.axes().yaxis.grid(b=True, which='major', linestyle='--')
if args.ylimit is not None:
	plt.ylim(0.0,args.ylimit)
plt.xlabel('Structure',fontweight='bold',fontsize='large')
plt.ylabel('Localisation Error',fontweight='bold',fontsize='large')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()
