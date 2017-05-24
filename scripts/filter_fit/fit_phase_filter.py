#!/usr/bin/python

# Imports
import os              # linesep
import sys             # argv
import numpy as np     # array, loadtxt
import scipy.stats	   # gamma distribution
import argparse as ap  # parser (for arguments)
import glob            # glob
import cv2             # OpenCV for opening videos
import math            # isnan
import CPBUtils as ut

def fit_phase_filter(track_directory,video_directory,parameters_file,outfilename,excludelist=[]):

	# Get a list of track files
	error_flag,trackfiles_ex = ut.getTracksList(track_directory,'.tk',excludelist)
	if error_flag:
		exit()

	# An empty list to store transitions
	transitions = []

	# Loop through the videos
	for filename in trackfiles_ex :

		# Read in the trackfile data as a table
		table = np.loadtxt(filename, skiprows=3)
		nFrames = int(table[-1][ut.tk_frameCol]) + 1

		# Attempt to open the video to find the frame rate
		vidname = os.path.splitext(os.path.basename(filename))[0]
		vidobj = cv2.VideoCapture(os.path.join(video_directory , vidname + '.avi'))

		frame_rate = vidobj.get(cv2.CAP_PROP_FPS)
		vidobj.release()

		# If this failed due to the silly OpenCV bug, read the database file instead
		if math.isnan(frame_rate):
			with open(os.path.join(video_directory,"frameratedatabase"), 'r') as openfile:
				for line in openfile:
					if line.split()[0] == vidname + '.avi' :
						frame_rate = float(line.split()[1])

		if math.isnan(frame_rate):
			print "Uanble to determine frame rate for video", vidname

		systole_deltas = []
		diastole_deltas = []

		# Loop through the frames (except the last)
		for f in range(0,nFrames-1) :

			# Check that the frame and the following frame are valid
			if int(table[f][ut.tk_labelledCol]) == 1 and int(table[f][ut.tk_presentCol]) == 1 and int(table[f+1][ut.tk_labelledCol]) == 1 and int(table[f+1][ut.tk_presentCol]) and float(table[f][ut.tk_cardiacPhaseCol]) >= 0.0 and float(table[f+1][ut.tk_cardiacPhaseCol]) >= 0.0 :

				# Change in phase
				delta = (float(table[f+1][ut.tk_cardiacPhaseCol]) - float(table[f][ut.tk_cardiacPhaseCol]))

				# Wrap to range 0 to 2pi
				delta %= 2.0*np.pi

				# Store normalised value
				norm_delta = delta*frame_rate
				#if norm_delta > 25.0 or norm_delta < 10.0 :
				#	print vidname, f, delta
				transitions.append(norm_delta)

				if float(table[f][ut.tk_cardiacPhaseCol]) > np.pi :
					diastole_deltas.append(norm_delta)
				else :
					systole_deltas.append(norm_delta)


	# Change list to np array
	transitions = np.array(transitions)

	# Fit the gamma distribution
	alpha,_,beta = scipy.stats.gamma.fit(transitions,floc=0.0)

	# Write the alpha and beta parameters into the out file
	# overriding the current contents of that line
	with open(parameters_file,'r') as param_file:
		lines = param_file.readlines()

	lines[4] = str(alpha) + " " + str(beta) + '\n'

	with open(outfilename,'w') as ofile:
		ofile.writelines(lines)


# Argument parsing when run as a script
if __name__ == '__main__':

	# Set up argument parser
	parser = ap.ArgumentParser(description='Fit a distribution to the phase transitions')
	parser.add_argument('track_directory',help="directory containing the track files")
	parser.add_argument('video_directory',help="directory containing the video files")
	parser.add_argument('model_parameters_file',help="name of the input file containing basic parameters")
	parser.add_argument('outfilename',help="name of the out file")
	parser.add_argument('--cross_val','-c',action='store_true',help='Create filter files for each fold in a leave-one-out cross-validation')
	parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
	args = parser.parse_args()

	if args.cross_val and args.exclude_list:
		print("ERROR: Cannot use cross_val option and provide an exclude list")
		sys.exit()

		# Make the function call
	if args.cross_val:
		patients_list = ut.getPatientsInTrackDirectory(args.track_directory)
		for patient in patients_list:
			fit_phase_filter(args.track_directory,args.video_directory,args.model_parameters_file,args.outfilename+'_ex'+patient,excludelist=[patient])
	else:
		fit_phase_filter(args.track_directory,args.video_directory,args.model_parameters_file,args.outfilename,excludelist=args.exclude_list)
