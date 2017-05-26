#!/usr/bin/python

import argparse as ap
import os
import glob
from subprocess import call

parser = ap.ArgumentParser(description='Perform several testing experiments defined by values in test and train files')
parser.add_argument('testing_binary',help="the binary file that will be invoked")
parser.add_argument('experiments_file',help="file containing a list of experiment names and testing parameters")
parser.add_argument('training_experiments_file',help="file used to train the models listing all the models to be tested")
parser.add_argument('video_directory',help="directory containing the video files")
parser.add_argument('models_directory',help="directory containing the trained model files")
parser.add_argument('results_directory',help="directory to contain the results files")
parser.add_argument('track_directory',help="directory containing the track files")
parser.add_argument('mask_directory',help="directory containing the mask files")
parser.add_argument('--ignore_existing','-i',action='store_true',help="do not retest a model if the results file already exists")
parser.add_argument('--filter_def_stems','-z',nargs='*',help="stems to use to look for filter files")
parser.add_argument('--filter_extension','-f',default='',help="extension for filter files")
parser.add_argument('--verbose','-v',action='store_true',help="Print commands executing them")
parser.add_argument('--display','-d',type=int,help='Display mode (as in the test exectuable)',default=0)
parser.add_argument('--num_trials','-n',type=int,help='Number of times to repeat each trial')

args = parser.parse_args()

if not os.path.isfile(args.testing_binary) :
	print "ERROR: binary file does not exist: " + args.testing_binary
	exit()

# Get a list of video files to test, and the corresponding patients and masks
vids_list = glob.glob(os.path.join(args.video_directory,'*.avi'))
short_vid_names_list = [os.path.basename(vid).split('.')[0] for vid in vids_list]
patient_list = [os.path.basename(vid).rsplit('_',1)[0] for vid in vids_list]
mask_list = [os.path.join(args.mask_directory , patient + '_mask.png') for patient in patient_list]
trackfile_list = [os.path.join(args.track_directory , vid + '.tk') for vid in short_vid_names_list]
radius_list = [open(trackfile).readlines()[2].rstrip().split()[-1] for trackfile in trackfile_list]

# Read the experiments file
with open(args.experiments_file,'r') as experiments_file :
	test_experiment_lines = [e.rstrip(os.linesep) for e in experiments_file if len(e) > 1]

# Read in list of models to use
with open(args.training_experiments_file,'r') as training_file :
	models_list = [e.split()[0] for e in training_file]

# Loop over experiments
for exp_line in test_experiment_lines:

	# Split the line into the model name and the parameters
	exp_name,exp_params = exp_line.split(maxsplit=1)

	# Make a directory for this experiment if it doesn't exist
	experiment_results_dir = os.path,join(args.results_directory,exp_name)
	if not os.path.exists(experiment_results_dir):
		os.makedirs(experiment_results_dir)

	# Loop over trained forest models
	for model_name in models_list:

		# Stem of the model file to use (without excluded patient)
		model_file_stem = os.path.join( args.models_directory , model_name , model_name)

		# Make a directory within the results directory for this models results
		this_model_results_directory = os.path.join(experiment_results_dir , model_name)
		if args.filter_def_stems is not None:
			this_model_results_directory = os.path.join(this_model_results_directory,'__'.join(map(os.path.basename,args.filter_def_stems)))
		if not os.path.exists(this_model_results_directory):
			os.makedirs(this_model_results_directory)

		# Loop over videos
		for full_vid_name,short_vid_name,patient,mask_file,radius,trackfile in zip(vids_list,short_vid_names_list,patient_list,mask_list,radius_list,trackfile_list) :

			# Model name
			model_file =  model_file_stem + '_ex' + patient

			# Filter file name
			if args.filter_def_stems is not None:
				filter_defs_list = ['-z'] + [ (f + '_ex' + patient + args.filter_extension) for f in args.filter_def_stems]
			else:
				filter_defs_list = []

			# Results file name
			results_file_stem = os.path.join(this_model_results_directory, 'results_' + short_vid_name)
			results_file_list = [results_file_stem] if args.num_trials is None else [results_file_stem + '_r' + str(n) for n in range(args.num_trials)]

			# Loop through all the required results files
			for results_file in results_file_list:

				# Check whether this file already exists
				if args.ignore_existing and os.path.exists(results_file):
					continue

				# Construct the command to call
				command_list = [args.testing_binary,'-v',full_vid_name,'-m',model_file,'-o',results_file,'-r',radius,'-d',str(args.display),'-g',trackfile,'-k',mask_file] + filter_defs_list + exp_params.split()

				# Print the command if in verbose mode
				if args.verbose:
					print ' '.join(command_list)

				# Call the command
				call(command_list)
