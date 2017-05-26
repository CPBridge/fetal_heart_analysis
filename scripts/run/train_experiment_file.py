#!/usr/bin/python

import argparse as ap
import os
from subprocess import call

parser = ap.ArgumentParser(description='Train a set of models')
parser.add_argument('training_binary',help="the binary file that will be invoked")
parser.add_argument('experiments_file',help="file containing a list of experiment names and training parameters")
parser.add_argument('video_directory',help="directory containing the video files")
parser.add_argument('dataset_directory',help="directory containing the dataset files")
parser.add_argument('dataset_name',help="stem of dataset name within directory")
parser.add_argument('models_directory',help="directory to contain the random forest models")
parser.add_argument('--ignore_existing','-i',action='store_true',help="do not retrain a model if the file already exists")
parser.add_argument('--verbose','-v',action='store_true',help="output the commands before calling them")
parser.add_argument('--use_structure_dataset','-s',action='store_true',help="append '_subs' to the datset names in order to use the strucure datasets")

args = parser.parse_args()

if not os.path.isfile(args.training_binary) :
	print "ERROR: binary file does not exist: " + args.training_binary
	exit()

# Read the patients file
with open(os.path.join(args.video_directory,'patients'),'r') as patients_file:
	patients = [l.rstrip(os.linesep) for l in patients_file]

# Read the experiments file
with open(args.experiments_file,'r') as experiments_file :
	experiment_lines = [e.rstrip(os.linesep) for e in experiments_file if len(e) > 1]

# Loop over experiments
for exp_line in experiment_lines:

	# Split the line into the model name and the parameters
	exp_name,exp_params = exp_line.split(maxsplit=1)

	# Make a directory for this experiment if it doesn't exist
	experiment_model_dir = os.path.join(args.models_directory , exp_name )
	if not os.path.exists(experiment_model_dir):
		os.makedirs(experiment_model_dir)

	# Loop over excluded patients
	for excluded_patient in patients :

		# Name of the dataset to use
		dataset_name = os.path.join(args.dataset_directory , args.dataset_name + '_ex' + excluded_patient)
		if args.use_structure_dataset:
			dataset_name += '_subs'

		# Model name
		model_name = experiment_model_dir + exp_name + '_ex' + excluded_patient

		# Check whether this file already exists
		if args.ignore_existing and os.path.exists(model_name + '.tr'):
			continue

		# Construct the command to call
		command_list = [args.training_binary,'-v',args.video_directory,'-o',model_name,'-d',dataset_name] + exp_params.split()

		# Output the command if necessary
		if args.verbose:
			print ' '.join(command_list)

		# Call the command
		call(command_list)
