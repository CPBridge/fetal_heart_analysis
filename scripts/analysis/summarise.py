#!/usr/bin/python

import os
import sys
import argparse as ap
from parse_output import parse_output
from glob import glob

default_track_dir = "/data/ChristosVids/tracks/"
default_radius_thresh = 0.25

# Parse command line arguments
parser = ap.ArgumentParser(description='Process filter outputs to produce summary files for an experiment')
parser.add_argument('test_experiment_file',help="List of testing experiments to summarise")
parser.add_argument('train_experiment_file',help="List of training experiments")
parser.add_argument('results_dir',help="Directory where the results are stored, in directories matching the experiment name")
parser.add_argument('--radius_threshold','-r',type=float,help="Radius threshold to use to decide on postive detections",default=default_radius_thresh)
parser.add_argument('--match_pattern','-m',help="Use this pattern to decide which files to include",default="results*")
parser.add_argument('--track_directory','-t',help="Directory containing the trackfiles",default=default_track_dir)
parser.add_argument('--structs_track_directory','-s',help="Directory containing the structure trackfiles",default='')
parser.add_argument('--weight_thresh','-w',help="If positive, use this weight threshold to determine whether a detection is made, if negative use the 'hidden' output",type=float,default=-1.0)
parser.add_argument('--output_name','-o',help="Specify the output summary file name (in the experiment directory unless otherwise stated)",default="summary")
parser.add_argument('--use_test_experiment_name','-e',action="store_true",help="If this flag is set, treat the test_experiment_file input as the experiment name itself, rather than a file containing them")
parser.add_argument('--use_train_experiment_name','-f',action="store_true",help="If this flag is set, treat the train_experiment_file input as the experiment name itself, rather than a file containing them")
parser.add_argument('--ground_truth_locations','-P',action="store_true",help='Use this if the experiments were carried out with using the ground truth locations for phase and orientation predictions')
parser.add_argument('--filtering_sub_folders','-z',action='store_true',help='Summarise each subdirectory of the listed train/test directories, relating to filtering files')
args = parser.parse_args()

header_str = "model video frames_processed correct_detections correct_rejections misses incorrect_detections incorrect_locations class_confusions obscured_detections obscured_rejections obscured_incorrect_locations obscured_class_confusions c00 c01 c02 c03 c10 c11 c12 c13 c20 c21 c22 c23 c30 c31 c32 c33 oc00 oc01 oc02 oc03 oc10 oc11 oc12 oc13 oc20 oc21 oc22 oc23 oc30 oc31 oc32 oc33 ori_error phase_error time_per_frame" + os.linesep

if args.use_test_experiment_name :
	test_experiment_names = [args.test_experiment_file]
else:
	with open(args.test_experiment_file,'r') as f:
		test_experiment_names = [l.split()[0] for l in f]

if args.use_train_experiment_name :
	train_experiment_names = [args.train_experiment_file]
else:
	with open(args.train_experiment_file,'r') as f:
		train_experiment_names = [l.split()[0] for l in f]


# Function to summarise one directory
def summarise_dir(exp_dir):

	summary_lines = [header_str]

	for results_file in glob( os.path.join(exp_dir,args.match_pattern) ) :
		if args.weight_thresh < 0.0 :
			summary_lines.append(parse_output(results_file, args.track_directory, None, 0, 0, 0.25, use_visible = True, include_misdetections=args.ground_truth_locations,structs_track_directory=args.structs_track_directory))
		else:
			summary_lines.append(parse_output(results_file, args.track_directory, None, 0, 0, 0.25, use_visible = False, posterior_threshold = args.weight_thresh, include_misdetections=args.ground_truth_locations,structs_track_directory=args.structs_track_directory))

	if os.path.sep in args.output_name :
		output_file = args.output_name
	else:
		output_file = os.path.join( exp_dir , args.output_name)

	with open(output_file,'w') as f:
		f.writelines(summary_lines)


# Loop through the experiments
for test_exp in test_experiment_names :

	for train_exp in train_experiment_names:

		if args.filtering_sub_folders:

			filter_folders = glob(os.path.join(args.results_dir , test_exp , train_exp , '*' , ''))
			for exp_dir in filter_folders:
				summarise_dir(exp_dir)

		else:

			summarise_dir(os.path.join( args.results_dir , test_exp  , train_exp , ''))
