#!/usr/bin/python

import CPBUtils as ut
import argparse as ap
import matplotlib.pyplot as plt           # plot

parser = ap.ArgumentParser(description='Plot Bar Graphs for Time Taken')
parser.add_argument('results_directory',help="directory containing the results, in folders according to model name")
parser.add_argument('train_experiment_definitions',help="file containing all of the training experiments to be displayed")
parser.add_argument('test_experiment_definitions',help="file containing all of the testing experiments to be displayed")
parser.add_argument('--std','-s',action='store_true',help="plot standard deviation bars")
parser.add_argument('--filter_file_name','-z',help='Results are found in a directory relating to the filterfile used with this name')

args = parser.parse_args()

if args.std:
	train,test,time,_,_,_,time_std,_,_,_ = ut.gatherAccuracyStatsTrainTest(args.train_experiment_definitions,args.test_experiment_definitions,args.results_directory,find_std=True,filtername=(None if args.filter_file_name is None else args.filter_file_name))
else:
	train,test,time,_,_,_ = ut.gatherAccuracyStatsTrainTest(args.train_experiment_definitions,args.test_experiment_definitions,args.results_directory,filtername=(None if args.filter_file_name is None else args.filter_file_name))

if len(train) != 1 :
	print "Can only deal with one train file"

index = range(len(test))
time = [t[0] for t in time]
bar_width = 0.5

fig = plt.figure(figsize=(10,6))
bar_basic = plt.bar(index, time, bar_width,color='cyan')
plt.xlabel('Calculation Method',fontweight='bold',fontsize='large')
plt.ylabel('Time per frame, ms',fontweight='bold',fontsize='large')
midpoints = [x+bar_width/2.0 for x in index]
plt.xticks(midpoints, test, rotation=90,fontsize='small',fontweight='bold')
plt.axes().yaxis.grid(b=True, which='major', linestyle='--')

if args.std:
	plt.errorbar(midpoints,time,yerr=time_std,color='r',linestyle='None')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()
