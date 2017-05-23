#!/usr/bin/python

# Imports
import os              # linesep
import sys             # argv
import numpy as np     # array, loadtxt
import glob            # glob
import math            # ceil
import argparse as ap  # parser (for arguments)
import cv2             # OpenCV
import CPBUtils as ut

def makeHeartDataset(track_directory,n_views,mask_directory=None,n_examples_per_view=100,outputfilename="dataset",excludelist=None,first_frame=0,overlap_frac=1.0,jitter=0,jitter_range=10):

	if not mask_directory :
		use_masks = False
	else :
		use_masks = True

	# Find track files
	if excludelist is None :
		excludelist = []
	_,trackfiles_ex = ut.getTracksList(track_directory,'.tk',excludelist)

	# Find the unique patients
	patients = [os.path.basename(f.rsplit('_',1)[0]) for f in trackfiles_ex]
	unique_patients = list(set(patients))
	n_patients = len(unique_patients)
	n_vids = len(trackfiles_ex)

	# Lists for number of frames of each view in each video
	n_frames_per_video_per_view = np.zeros((n_vids,n_views),int)

	# Lists for the number of frames of each view in each patient
	n_frames_per_patient_per_view = np.zeros((n_patients,n_views),int)

	# List for the patient id of each video
	patient_id_per_video = np.zeros(n_vids,int)

	# Loop through trackfiles
	for vd,filename in enumerate(trackfiles_ex):

		# The patient id for this video
		patient_id_per_video[vd] = unique_patients.index(patients[vd])

		# Read in the track data for this video
		table,image_dims,_,radius = ut.readHeartTrackFile(filename)
		radius = int(radius)

		# Try and read the mask for this video
		if use_masks :
			mask_string = os.path.join( mask_directory , os.path.basename(filename).rsplit('_',1)[0] + '_mask.png')
			mask = cv2.imread(mask_string,0)
			if mask is None :
				print 'Could not open mask file', mask_string
			edgedistimage = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

		# Create new table with just frames where the heart is present
		for i in range(first_frame,len(table)) :
			distance_ok = (not use_masks) or (edgedistimage[int(table[i,ut.tk_yposCol]),int(table[i,ut.tk_xposCol])] > radius*overlap_frac)
			if (table[i,ut.tk_presentCol] == 1) and distance_ok :
				vw = int(table[i,ut.tk_viewCol] - 1)
				n_frames_per_video_per_view[vd,vw] += 1
				n_frames_per_patient_per_view[patient_id_per_video[vd],vw] += 1

		# Store the number of each view in this video and for this patient
		#for vw in range(0,n_views):
		#	n_frames_per_video_per_view[vd,vw] = (presentTable[:,ut.tk_viewCol] == vw+1).sum()
		#	n_frames_per_patient_per_view[patient_id_per_video[vd],vw] += n_frames_per_video_per_view[vd,vw]

	frames_selected_per_patient_per_view = np.zeros([n_patients,n_views],int)

	# Choose how to divide up the training instances among the patients
	for vw in range(0,n_views):

		frames_to_select = n_examples_per_view

		# Check that the number of requested frames is not greater than the number that there actually is
		if (n_examples_per_view > n_frames_per_patient_per_view[:,vw].sum() ) :
			print "Warning: There are insufficient frames for view number", vw+1, ", size of dataset will be reduced"
			frames_selected_per_patient_per_view[:,vw] = n_frames_per_patient_per_view[:,vw]
			break

		frames_unselected_per_patient = n_frames_per_patient_per_view[:,vw]

		while frames_to_select > 0 :

			# Number of patients with frames left
			nonzerocounts = (frames_unselected_per_patient != 0).sum()

			# Select the same number from each patient
			if (frames_to_select >= nonzerocounts) :
				frames_per_patient = frames_to_select // nonzerocounts

				for p in range(0,n_patients) :
					if (frames_unselected_per_patient[p] - frames_per_patient >= 0) :
						frames_unselected_per_patient[p] -= frames_per_patient
						frames_to_select -= frames_per_patient
						frames_selected_per_patient_per_view[p,vw] += frames_per_patient
					else :
						frames_to_select -= frames_unselected_per_patient[p]
						frames_selected_per_patient_per_view[p,vw] += frames_unselected_per_patient[p]
						frames_unselected_per_patient[p] = 0

			# Select randomly from patients
			else :
				nonzero_patients = [p for p,val in enumerate(frames_unselected_per_patient) if val > 0]
				chosen_patients = np.random.choice(nonzero_patients,frames_to_select,False)

				for p in chosen_patients :
					frames_selected_per_patient_per_view[p,vw] += 1
					frames_to_select -= 1

	frames_selected_per_video_per_view = np.zeros([n_vids,n_views],int)

	# Choose how to divide up the training instances among the videos
	for p in range(0,n_patients) :
		for vw in range(0,n_views) :
			frames_to_select = frames_selected_per_patient_per_view[p,vw]

			frames_unselected_per_video = np.copy(n_frames_per_video_per_view[:,vw])
			frames_unselected_per_video[patient_id_per_video != p] = 0

			while frames_to_select > 0 :

				# Number of videos with frames left
				nonzerocounts = (frames_unselected_per_video != 0).sum()
				if nonzerocounts == 0 :
					break

				if (frames_to_select >= nonzerocounts) :
					frames_per_video = frames_to_select // nonzerocounts

					for vd in range(0,n_vids) :
						if (frames_unselected_per_video[vd] - frames_per_video >= 0) :
							frames_unselected_per_video[vd] -= frames_per_video
							frames_to_select -= frames_per_video
							frames_selected_per_video_per_view[vd,vw] += frames_per_video
						else :
							frames_to_select -= frames_unselected_per_video[vd]
							frames_selected_per_video_per_view[vd,vw] += frames_unselected_per_video[vd]
							frames_unselected_per_video[vd] = 0

				# Select randomly from patients
				else :
					nonzero_videos = [vd for vd,val in enumerate(frames_unselected_per_video) if val > 0 ]
					chosen_videos = np.random.choice(nonzero_videos,frames_to_select,False)

					for vd in chosen_videos :
						frames_selected_per_video_per_view[vd,vw] += 1
						frames_to_select -= 1

	# Open an output file
	outfile = open(outputfilename,'w')
	outfile.write('filename frameno radius centrey centrex label ori phase' + os.linesep)
	outfile.write("background 4CHAM LVOT 3V" + os.linesep)
	outfile.write(str(2*frames_selected_per_video_per_view.sum()*(jitter+1)) + os.linesep)

	# Loop through each video and select random frames to go into the dataset
	# Loop through trackfiles
	for vd,filename in enumerate(trackfiles_ex):

		# Name of the video
		vidname = os.path.basename(os.path.splitext(filename)[0]) + '.avi'

		# Read in the track data for this video
		table,image_dims,_,radius = ut.readHeartTrackFile(filename)
		radius = int(radius)
		width = image_dims[0]
		height = image_dims[1]

		# Try and read the mask for this video
		if use_masks :
			mask_string = os.path.join( mask_directory , vidname.rsplit('_',1)[0] + '_mask.png')
			mask = cv2.imread(mask_string,0)
			if mask is None :
				print 'Could not open mask file', mask_string
			edgedistimage = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

		# Cycle through views and choose a random selection of frames for each view
		frames_chosen = np.empty(0,int)
		for vw in range(0,n_views):
			# To be availabe a frame must have the heart present, in the correct view and have a frame number greater than "first frame"
			frames_available = []

			for i in range(0,len(table)) :
				distance_ok = (not use_masks) or (edgedistimage[int(table[i,ut.tk_yposCol]),int(table[i,ut.tk_xposCol])] > radius*overlap_frac)
				if (int(table[i,ut.tk_viewCol]) == vw+1) and (int(table[i,ut.tk_presentCol]) == 1) and (int(table[i,ut.tk_frameCol]) >= first_frame) and distance_ok :
					frames_available.append(int(table[i,ut.tk_frameCol]))

			frames_available =  np.array(frames_available,int)

			if len (frames_available) > 0 :
				# Choose some random frames from this view to add to the list
				newframes = np.array(np.random.choice(frames_available,frames_selected_per_video_per_view[vd,vw],False),int)
				frames_chosen = np.concatenate([frames_chosen, newframes])

		# Sort into ascending order (this allows video to be read in quicker at train time)
		frames_chosen.sort()

		for f in frames_chosen :

			xpos = table[f,ut.tk_xposCol]
			ypos = table[f,ut.tk_yposCol]
			vw = table[f,ut.tk_viewCol]

			# Write this positive example to file
			outfile.write(vidname + ' ')
			outfile.write(str(f) + ' ')
			outfile.write(str(radius) + ' ')
			outfile.write(str(int(ypos)) + ' ')
			outfile.write(str(int(xpos)) + ' ')
			outfile.write(str(int(vw)) + ' ')
			outfile.write(str(int(table[f,ut.tk_oriCol])) + ' ')
			outfile.write(str(float(table[f,ut.tk_cardiacPhaseCol])) + os.linesep)

			# If required, add some jittered versions
			if jitter > 0 :
				jitter_options = []
				for x_off in range(-jitter_range,jitter_range+1) :
					for y_off in range(-jitter_range,jitter_range+1) :
						if ((x_off != 0) or (y_off != 0)) :
							jitx = xpos + x_off
							jity = ypos + y_off
							if edgedistimage[jity,jitx] > radius*overlap_frac :
								jitter_options.append([jitx,jity])

				if jitter > len(jitter_options) :
					print "Error, there are not enough jitter options within the range provided to give the requested number of jittered positions "
					print edgedistimage[ypos,xpos], radius*overlap_frac
				options_chosen = np.random.choice(range(len(jitter_options)),jitter,False)

				for op in options_chosen :
					outfile.write(vidname + ' ')
					outfile.write(str(f) + ' ')
					outfile.write(str(radius) + ' ')
					outfile.write(str(int(jitter_options[op][1])) + ' ')
					outfile.write(str(int(jitter_options[op][0])) + ' ')
					outfile.write(str(int(vw)) + ' ')
					outfile.write(str(int(table[f,ut.tk_oriCol])) + ' ')
					outfile.write(str(float(table[f,ut.tk_cardiacPhaseCol])) + os.linesep)


			# Create a matching number of random background examples
			for _ in range(jitter + 1) :
				# Select a random negative example from this frame that is within the permissible area
				while True :
					bgxpos = np.random.randint(int(math.ceil(radius))+1,int(math.floor(width-radius))-1)
					bgypos = np.random.randint(int(math.ceil(radius))+1,int(math.floor(height-radius))-1)

					heart_dist = ((bgxpos-xpos)**2 + (bgypos-ypos)**2)**0.5

					if use_masks :
						edgedist = edgedistimage[bgypos,bgxpos]
					else :
						edgedist = 1000000

					if heart_dist/radius > 0.3 and edgedist - 2 > radius*overlap_frac :
						break

				# Generate a random orientation for this background example
				bg_ori = np.random.randint(0,359)

				# Write this background example to file
				outfile.write(vidname + ' ')
				outfile.write(str(f) + ' ')
				outfile.write(str(radius) + ' ')
				outfile.write(str(int(bgypos)) + ' ')
				outfile.write(str(int(bgxpos)) + ' ')
				outfile.write('0 ') # view = background
				outfile.write(str(int(bg_ori)) + ' ') # ori
				outfile.write('0.0 ' + os.linesep) # phase (irrelevant)

	outfile.close()

def makeHeartCrossValidationDatasets(track_directory,n_views,mask_directory=None,n_examples_per_view=100,outputfilename="dataset",first_frame=0,overlap_frac=1.0,jitter=0,jitter_range=10):

	# Get a list of all the trackfiles
	_,trackfiles = ut.getTracksList(track_directory,'.tk')

	# Find the unique patients in this list
	patients = [os.path.basename(f.rsplit('_',1)[0]) for f in trackfiles]
	unique_patients = list(set(patients))

	# Create one dataset excluding each of the patients
	for excluded_patient in unique_patients:
		this_dataset_name = outputfilename + '_ex' + excluded_patient
		makeHeartDataset(track_directory,n_views,mask_directory,n_examples_per_view,this_dataset_name,[excluded_patient],first_frame,overlap_frac,jitter,jitter_range)



# Execute as a script by parsing command line arguments and calling the above function
if __name__ == "__main__" :

	# Parse command line arguments
	parser = ap.ArgumentParser(description='Create a dataset from a set of trackfiles')
	parser.add_argument('track_directory',help="directory containing the track files")
	parser.add_argument('--mask_directory','-m',help="directory containing the image masks",default='')
	parser.add_argument('--n_views','-v',type=int,help="number of view types labelled (excluding background)")
	parser.add_argument('--n_examples_per_view','-n',type=int,help="number of examples per view",default=100)
	parser.add_argument('--outfilename','-o',help="name of the output file",default="dataset")
	parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
	parser.add_argument('--cross_val','-c',action='store_true',help="produce multiple datasets, one with each patient excluded")
	parser.add_argument('--first_frame','-f',type=int,help="avoid frames before this frame",default=0)
	parser.add_argument('--jitter_number','-j',type=int,help="number of times to jitter each position",default=0)
	parser.add_argument('--jitter_range','-r',type=int,help="maximum offset for jitter",default=10)
	parser.add_argument('--overlap_frac','-l',type=float,help="fraction of the radius that must be within the masked area for a background sample",default=1.0)
	args = parser.parse_args()

	# Check the inputs don't conflict
	if args.cross_val and args.exclude_list:
		print("ERROR: Cannot use cross_val option and provide an exclude list")
		sys.exit()

	if args.cross_val :
		# Call function to make several datasets
		makeHeartCrossValidationDatasets(args.track_directory,args.n_views,args.mask_directory,args.n_examples_per_view,args.outfilename,args.first_frame,args.overlap_frac,args.jitter_number,args.jitter_range)
	else:
		# Call the function to make a single dataset
		makeHeartDataset(args.track_directory,args.n_views,args.mask_directory,args.n_examples_per_view,args.outfilename,args.exclude_list,args.first_frame,args.overlap_frac,args.jitter_number,args.jitter_range)
