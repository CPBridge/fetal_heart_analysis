#!/usr/bin/python
import sys             # stdin
import argparse as ap  # parser (for arguments)
import numpy as np     # loadtxt, random.choice
import shutil          # copyfile
import os              # path.isfile
import subprocess      # subprocess
import math            # pi, cos
import CPBUtils as ut  # coloumn constants

# Number of classes (including background)
n_classes = 4

def parse_output(resultsfile, track_directory,  inputdataset, n_negsamples, n_possamples, radius_threshold, posterior_threshold=0.0, use_visible=False, summary=False, include_misdetections=False, structs_track_directory='') :

	gathering = (inputdataset is not None and outputdataset is not None_)

	# Lists to record candidates for new training data
	hard_negs = []
	hard_pos = []

	# Counts of true positives etc
	n_correct_detections = 0
	n_correct_rejections = 0
	n_incorrect_detections = 0
	n_misses = 0
	n_incorrect_locations = 0
	n_class_confusions = 0
	n_obscured_rejections = 0
	n_obscured_class_confusions = 0
	n_obscured_detections = 0
	n_obscured_incorrect_locations = 0
	confusion_matrix = np.zeros([n_classes,n_classes],int)
	obscured_confusion_matrix = np.zeros([n_classes,n_classes],int)

	# Load the results data
	results_data,vidname,modelname,radius_det,time_per_frame = ut.getResultsFileDataAsRows(resultsfile)

	# Work out what variables we should be measuring
	measure_phase = "phase" in results_data[0]
	if measure_phase :
		phase_error = 0.0
	measure_ori = "ori" in results_data[0]
	if measure_ori :
		ori_error = 0.0

	find_structures = ('structs' in results_data[0]) and (len(structs_track_directory) > 0)
	if(find_structures):
		structure_names = [name for name in results_data[0]['structs']]

	# Load the relevant trackfile
	tracktable,_,_,radius_tk = ut.readHeartTrackFileAsRows(os.path.join(track_directory , vidname + '.tk'))
	distance_threshold = radius_tk*radius_threshold
	n_labelled_frames = len([1 for i in tracktable if i['labelled'] ])

	# Load structure trackfile if needed
	if(find_structures):
		structs_track_data = ut.readMultipleStructuresTrackAsDicts( os.path.join(structs_track_directory , vidname + '.stk') , structure_names )

		# Initialise containers for results
		struct_correct_detections = { name : 0 for name in structure_names }
		struct_misses = { name : 0 for name in structure_names }
		struct_correct_rejections = { name : 0 for name in structure_names }
		struct_incorrect_detections = { name : 0 for name in structure_names }
		struct_distance_error = { name : 0.0 for name in structure_names }

	# Check that the radii are similar
	if abs(radius_tk - radius_det) > 10 :
		print "Warning: detection radius (", radius_det, ") and annotated radius (", radius_tk, ") are significantly different"

	# Read in the frame numbers that are already in the dataset
	if gathering :
		with open(inputdataset,'r') as datasetfile:
			frames_positive_in_dataset = [ int(l.split()[ut.ds_frameCol]) for l in datasetfile if l.split()[ut.ds_vidCol] == vidname + '.avi' and int(l.split()[ut.ds_viewCol]) > 0 ]

	# Loop through the detection lines
	for results_row in results_data :

		# Current frame number
		frame_det = results_row['frame']

		# Get the line relevant to this frame from the track data
		track_row = tracktable[frame_det]
		if frame_det != track_row['frame'] :
			print "Error: non-matching tracktable entries!"
		y_tk = track_row['y']
		x_tk = track_row['x']
		class_tk = track_row['class']
		labelled_tk = track_row['labelled']
		present_tk = track_row['present']
		ori_tk = track_row['ori']
		phase_tk = track_row['phase']

		post_det = results_row['post']
		visible_det = results_row['visible']
		if measure_phase :
			phase_det = results_row['phase']
		if measure_ori :
			ori_det = results_row['ori']
		if find_structures:
			structs_det = results_row['structs']
			structs_track = structs_track_data[frame_det]

		if not labelled_tk :
			# Don't have a label for this - do nothing
			pass
		elif ( (not use_visible and (post_det < posterior_threshold)) or (use_visible and not visible_det ) ) :
			# No detection made
			if present_tk == 1 :

				# A false negative (miss), record as a hard positive if it isn't already in the dataset
				n_misses += 1
				confusion_matrix[class_tk,0] += 1

				if gathering and frame_det not in frames_positive_in_dataset :
					hard_pos.append(str(frame_det) + ' ' +  str(int(radius_tk)) + ' ' + str(y_tk) + ' ' + str(x_tk) + ' ' + str(class_tk) + ' ' + str(int(ori_tk*(180.0/math.pi))) + " " + str(phase_tk))

				if include_misdetections:
					if measure_ori :
						ori_error += 0.5*(1.0-math.cos(ori_tk-ori_det))
					if measure_phase :
						phase_error += 0.5*(1.0-math.cos(phase_tk-phase_det))

			elif present_tk == 2 :
				# A false negative (miss), record as a hard positive if it isn't already in the dataset
				n_obscured_rejections += 1
				obscured_confusion_matrix[class_tk,0] += 1
			else :
				# A true negative - all is well
				n_correct_rejections += 1
				confusion_matrix[0,0] += 1
		else :
			# Something has been detected
			# Get the detection information from the input stream
			y_det = results_row['y']
			x_det = results_row['x']
			class_det = results_row['class']

			if present_tk in [1,2] :
				position_correct = (float( (y_tk - y_det)**2 + (x_tk - x_det)**2 )**0.5 < distance_threshold)
				view_correct = (class_tk == class_det)

				if position_correct :
					if view_correct :
						if present_tk == 2 :
							n_obscured_detections += 1
							confusion_matrix[class_tk,class_det] += 1
						else :
							# A true positive, all is well
							n_correct_detections += 1
							confusion_matrix[class_tk,class_det] += 1

							if measure_ori :
								ori_error += 0.5*(1.0-math.cos(ori_tk-ori_det))
							if measure_phase :
								phase_error += 0.5*(1.0-math.cos(phase_tk-phase_det))

							if find_structures:
								for struct in structure_names:
									if structs_track[struct]['labelled']:
										if (structs_det[struct]['visible'] == 1) and (structs_track[struct]['present'] == 1):
											# A correct detection of this structure
											struct_correct_detections[struct] += 1
											struct_distance_error[struct] += (float( (structs_track[struct]['y'] - structs_det[struct]['y'])**2 + (structs_track[struct]['x'] - structs_det[struct]['x'])**2 )**0.5)/radius_tk

										elif (structs_det[struct]['visible'] == 0) and (structs_track[struct]['present'] == 1):
											# A miss for this structure
											struct_misses[struct] += 1

										elif (structs_det[struct]['visible'] == 0) and (structs_track[struct]['present'] in [0,2]):
											# A correct rejection for this structure
											struct_correct_rejections[struct] += 1

										elif (structs_det[struct]['visible'] == 1) and (structs_track[struct]['present'] in [0,2]):
											# An incorrect detection this structure
											struct_incorrect_detections[struct] += 1

					else :
						# Detected the heart but mistaken one view for another
						# place in the dataset candidates if not there already
						if present_tk == 2 :
							n_obscured_class_confusions += 1
							obscured_confusion_matrix[class_tk,class_det] += 1
						else :
							n_class_confusions += 1

							# Record this is in the confusion matrix
							confusion_matrix[class_tk,class_det] += 1

							if gathering and frame_det not in frames_positive_in_dataset :
								hard_pos.append(str(frame_det) + " " + str(int(radius_tk)) + " " + str(y_tk) + " " + str(x_tk) + " " + str(class_tk) + " " + str(int(ori_tk*(180.0/math.pi))) + " " + str(phase_tk))

							if include_misdetections:
								if measure_ori :
									ori_error += 0.5*(1.0-math.cos(ori_tk-ori_det))
								if measure_phase :
									phase_error += 0.5*(1.0-math.cos(phase_tk-phase_det))

				else : # i.e. position is not correct
					# Detected in the wrong place - have a hard negative and potentially a hard positive
					if present_tk == 2 :
						n_obscured_incorrect_locations += 1
						obscured_confusion_matrix[class_tk,0] += 1
					else:
						n_incorrect_locations += 1
						confusion_matrix[class_tk,0] += 1

						if gathering:
							hard_negs.append(str(frame_det) + " " + str(int(radius_tk)) + " " + str(y_det) + " " + str(x_det) + " 0 0 0.0")
							if frame_det not in frames_positive_in_dataset :
								hard_pos.append(str(frame_det) + " " + str(int(radius_tk)) + " " + str(y_tk) + " " + str(x_tk) + " " + str(class_tk) + " " + str(int(ori_tk*(180.0/math.pi))) + " " + str(phase_tk))

						if include_misdetections:
							if measure_ori :
								ori_error += 0.5*(1.0-math.cos(ori_tk-ori_det))
							if measure_phase :
								phase_error += 0.5*(1.0-math.cos(phase_tk-phase_det))

			else: # i.e. present_tk is zero
				# A false positive, record as a hard negative
				n_incorrect_detections += 1
				if gathering :
					hard_negs.append(str(frame_det) + " " + str(int(radius_tk)) + " " + str(y_det) + " " + str(x_det) + " 0 0 0.0")
				# Record this is in the confusion matrix
				confusion_matrix[0,class_det] += 1

	# Process and store the additional dataset lines if required
	if gathering:
		# Create a copy of the input dataset
		if not os.path.isfile(outputdataset) :
			shutil.copyfile(inputdataset,outputdataset)

		# Choose a subset of the candidates
		if len(hard_pos) > n_possamples :
			hard_pos_chosen = np.random.choice(hard_pos,n_possamples,replace=False)
		else :
			hard_pos_chosen = hard_pos

		if len(hard_negs) > n_negsamples :
			hard_negs_chosen = np.random.choice(hard_negs,n_negsamples,replace=False)
		else :
			hard_negs_chosen = hard_negs

		# Write the new samples on the end of the existing dataset
		with open(outputdataset,'a') as outputfile :
			for sample in hard_pos_chosen :
				outputfile.write(vidname + '.avi ' + sample + os.linesep)

			for sample in hard_negs_chosen :
				outputfile.write(vidname + '.avi ' + sample + os.linesep)

		# Run the bash script to fix the dataset
		subprocess.call(os.path.dirname(os.path.realpath(__file__)) + "../datasets/fixdataset.sh " + outputdataset, shell=True)

	# NB this may be fewer than the number of frames in the video because, if motion features are used, they are
	# not available in the first frame
	total_processed_frames = n_correct_detections + n_correct_rejections + n_incorrect_detections + n_misses + n_incorrect_locations + n_class_confusions + n_obscured_rejections + n_obscured_class_confusions	+ n_obscured_detections	+ n_obscured_incorrect_locations

	# The number of frames over which the phase and orientation error values were
	# accumulated
	n_phase_ori_frames = ( (n_correct_detections + n_class_confusions + n_misses + n_incorrect_locations) if include_misdetections else n_correct_detections)


	if measure_phase:
		if n_phase_ori_frames == 0 :
			phase_str = "nan"
		else :
			phase_str = str(phase_error/n_phase_ori_frames)
	else:
		phase_str = "n/a"

	if measure_ori:
		if n_phase_ori_frames == 0 :
			ori_str = "nan"
		else :
			ori_str = str(ori_error/n_phase_ori_frames)
	else:
		ori_str = "n/a"

	if find_structures:
		struct_distance_strings = { struct : ('nan' if struct_correct_detections[struct] == 0 else str(struct_distance_error[struct]/struct_correct_detections[struct])) for struct in structure_names }
		struct_str = ' '.join([ ' '.join([struct, str(struct_correct_detections[struct]), str(struct_correct_rejections[struct]), str(struct_misses[struct]), str(struct_incorrect_detections[struct]), struct_distance_strings[struct]])
		 for struct in structure_names])
	else:
		struct_str = ''

	# Print the results
	if summary :
		print os.linesep,"Results: ", os.linesep
		print "Of",n_labelled_frames,"labelled frames, there were"
		print "  ",n_correct_detections, "correct detections"
		print "  ",n_correct_rejections, "correct rejections"
		print "  ",n_misses, "misses"
		print "  ",n_incorrect_detections, "incorrect detections"
		print "  ",n_incorrect_locations, "incorrect locations (including correct classes)"
		print "  ",n_class_confusions, "class confusions (occuring at the right place)"
		print "  ",n_obscured_detections, "obscured detections"
		print "  ",n_obscured_rejections, "obscured rejections"
		print "  ",n_obscured_incorrect_locations, "obscured incorrect locations"
		print "  ",n_obscured_class_confusions,"obscured class confusions"
		print "  totalling", total_processed_frames
		print "The confusion matrix was: ", os.linesep, confusion_matrix
		print "The obscured confusion matrix was: ", os.linesep, obscured_confusion_matrix
		if measure_ori:
			print "The average orientation error (over the true positive frames), was ", ori_error/n_phase_ori_frames
		if measure_phase:
			print "The average phase error (over the true positive frames), was ", phase_error/n_phase_ori_frames
		if find_structures:
			for struct in structure_names:
				print "The results for structure", struct, "were"
				print "    ",struct_correct_detections[struct], "correct detections"
				print "    ",struct_correct_rejections[struct], "correct rejections"
				print "    ",struct_misses[struct], "misses"
				print "    ",struct_incorrect_detections[struct], "incorrect detections"
				print "    ",struct_distance_strings[struct],"average normalised distance error"
		print "The average time per frame was ", time_per_frame

	summary_string = (modelname + " " +
		vidname+ " " +
		str(total_processed_frames) + " " +
		str(n_correct_detections) + " " +
		str(n_correct_rejections) + " " +
		str(n_misses) + " " +
		str(n_incorrect_detections) + " " +
		str(n_incorrect_locations) + " " +
		str(n_class_confusions) + " " +
		str(n_obscured_detections) + " " +
		str(n_obscured_rejections) + " " +
		str(n_obscured_incorrect_locations) + " " +
		str(n_obscured_class_confusions) + " " +
		' '.join(str(confusion_matrix).replace('[','').replace(']','').replace(os.linesep,'').split()) + " " +
		' '.join(str(obscured_confusion_matrix).replace('[','').replace(']','').replace(os.linesep,'').split()) + " " +
		ori_str + " " +
		phase_str + " " +
		str(time_per_frame) + " " +
		struct_str +
		os.linesep)


	return summary_string


# Execute as a script by parsing command line arguments and calling the above function
if __name__ == "__main__" :

	parser = ap.ArgumentParser(description='Use the output from a detector to gather hard negatives for an existing dataset')
	parser.add_argument('results_file',help="file containing filter output to process")
	parser.add_argument('track_directory',help="directory containing the track file")
	parser.add_argument('n_negsamples',type=int,help="maximum number of new negative samples to choose")
	parser.add_argument('n_possamples',type=int,help="maximum number of new positive samples to choose")
	parser.add_argument('radius_threshold',type=float,help="permissible distance from centre to be considered correct (as fraction of radius)")
	parser.add_argument('--posterior_threshold','-q',type=float,help="threshold on the posterior at which to declare a detection")
	parser.add_argument('--use_visible','-v',action="store_true",help="use visible/hidden variable rather than threshold to determine detection")
	parser.add_argument('--inputdataset','-i',help="name of the input dataset file")
	parser.add_argument('--outputdataset','-o',help="name of the output dataset file")
	parser.add_argument('--summary','-s',help="print a summary of the findings to the terminal",action="store_true")
	parser.add_argument('--structs_track_directory','-t',help="directory containing the structure track file",default='')
	parser.add_argument('--include_misdetections','-m',action="store_true",help='accumulate phase and orientation error values over all frames for which the ground truth has a positive label')

	args = parser.parse_args()

	parse_output(args.results_file, args.track_directory,  args.inputdataset, args.n_negsamples, args.n_possamples, args.radius_threshold, posterior_threshold=args.posterior_threshold, use_visible=args.use_visible, summary=args.summary, include_misdetections=args.include_misdetections, structs_track_directory=args.structs_track_directory)
