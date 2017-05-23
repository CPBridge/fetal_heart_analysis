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


def makeSubsDataset(track_directory,heart_track_directory,radius_frac,substructure_file,view_index=-1,mask_directory=None,n_examples_per_struct=100,outputfilename="dataset",excludelist=None,first_frame=0,overlap_frac=1.0,list_heart_radius=False) :

	# Check that a mask directory was supplied
	if not mask_directory :
		print "ERROR: No mask directory supplied"
		return None

	# Read in the substructure list
	if not os.path.isfile(substructure_file):
		print "ERROR: The supplied substructure_file " + substructure_file + " does not exist!"
		return None
	substructurenames,viewsPerStructure,_,_,_ = ut.readSubstructureList(substructure_file)
	n_structures = len(substructurenames)

	# Find track files
	if excludelist is None :
		excludelist = []
	_,trackfiles_ex = ut.getTracksList(track_directory,'.stk',excludelist)

	if view_index == 0 :
		print "ERROR, Choosing a view label of 0 (background) doesn't make sense..."

	# Find the unique patients
	patients = [os.path.basename(f.rsplit('_',1)[0]) for f in trackfiles_ex]
	uniquePatients = list(set(patients))
	nPatients = len(uniquePatients)
	nVids = len(trackfiles_ex)

	# Lists for number of frames of each view in each video
	nFramesperVideoperStruct = np.zeros((nVids,n_structures),int)

	# Lists for the number of frames of each structure in each patient
	nFramesperPatientperStruct = np.zeros((nPatients,n_structures),int)

	# List for the patient id of each video
	patientidperVideo = np.zeros(nVids,int)

	# Loop through trackfiles to count the number of available frames in each
	for vd,filename in enumerate(trackfiles_ex):

		vidname = os.path.basename(os.path.splitext(filename)[0]) + '.avi'

		# The patient id for this video
		patientidperVideo[vd] = uniquePatients.index(patients[vd])

		# Read the second and third lines of the track file to get the video dimensions
		# and the detection radius
		heart_filename = os.path.join( heart_track_directory , os.path.basename(os.path.splitext(filename)[0]) + '.tk')
		heart_table,_,_,heart_radius = ut.readHeartTrackFile(heart_filename)

		# The radius at which to perform detection is a specified fraction of the
		# heart radius
		radius = heart_radius*radius_frac

		# Try and read the mask for this video
		mask_string = os.path.join( mask_directory , vidname.rsplit('_',1)[0] + '_mask.png')
		mask = cv2.imread(mask_string,0)
		if mask is None :
			print 'Could not open mask file', mask_string
		edgedistimage = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

		for s,struct in enumerate(substructurenames):
			# Read in the track data for this video
			table = ut.readSubstructure(filename,struct)

			if table is None:
				print "No data found for substructure", struct, "in track file", filename

			# Create new table with just frames where this structure is present
			framesAvailable = [
			int(table[i,ut.stk_frameCol]) for i in range(0,len(table))
			if ((int(table[i,ut.stk_presentCol]) == 1)
			and (int(table[i,ut.stk_frameCol]) >= first_frame)
			and ( edgedistimage[int(table[i,ut.stk_yposCol]),int(table[i,ut.stk_xposCol])] - 2 > radius*overlap_frac)
			and ( (view_index < 0) or (int(heart_table[i,ut.tk_viewCol]) == view_index) )
			and (int(heart_table[i,ut.tk_viewCol]) in viewsPerStructure[s]) )
			]

			# Store the number of each view in this video and for this patient
			nFramesperVideoperStruct[vd,s] = len(framesAvailable)
			nFramesperPatientperStruct[patientidperVideo[vd],s] += nFramesperVideoperStruct[vd,s]

	# Find out which structures were actually found in the dataset for
	# this view, call these the active structures
	if view_index > 0 :
		active_structures = [s for s in range(0,n_structures) if nFramesperPatientperStruct[:,s].sum() > 0]
		n_structures = len(active_structures)
		nFramesperVideoperStruct = nFramesperVideoperStruct[:,active_structures]
		nFramesperPatientperStruct = nFramesperPatientperStruct[:,active_structures]
		substructurenames = [substructurenames[s] for s in active_structures]

	framesSelectedperPatientperStruct = np.zeros([nPatients,n_structures],int)

	# Choose how to divide up the training instances among the patients
	for s in range(0,n_structures):

		framesToSelect = n_examples_per_struct

		if( (view_index < 0) and nFramesperPatientperStruct[:,s].sum() == 0 ) :
			# Silently continue to the next structure
			continue

		# Check that the number of requested frames is not greater than the number that there actually is
		if (n_examples_per_struct > nFramesperPatientperStruct[:,s].sum() ) :
			print "Warning: There are insufficient frames for structure", substructurenames[s], ", size of dataset will be reduced"
			framesSelectedperPatientperStruct[:,s] = nFramesperPatientperStruct[:,s]
			continue

		framesUnselectedperPatient = nFramesperPatientperStruct[:,s]

		while framesToSelect > 0 :

			# Number of patients with frames left
			nonzerocounts = (framesUnselectedperPatient != 0).sum()

			# Select the same number from each patient
			if (framesToSelect >= nonzerocounts) :
				framesperPatient = framesToSelect // nonzerocounts

				for p in range(0,nPatients) :
					if (framesUnselectedperPatient[p] - framesperPatient >= 0) :
						framesUnselectedperPatient[p] -= framesperPatient
						framesToSelect -= framesperPatient
						framesSelectedperPatientperStruct[p,s] += framesperPatient
					else :
						framesToSelect -= framesUnselectedperPatient[p]
						framesSelectedperPatientperStruct[p,s] += framesUnselectedperPatient[p]
						framesUnselectedperPatient[p] = 0

			# Select randomly from patients
			else :
				nonzeroPatients = [p for p,val in enumerate(framesUnselectedperPatient) if val > 0]
				chosenPatients = np.random.choice(nonzeroPatients,framesToSelect,False)

				for p in chosenPatients :
					framesSelectedperPatientperStruct[p,s] += 1
					framesToSelect -= 1

	framesSelectedperVideoperStruct = np.zeros([nVids,n_structures],int)

	# Choose how to divide up the training instances among the videos
	for p in range(0,nPatients) :
		for s in range(0,n_structures) :
			framesToSelect = framesSelectedperPatientperStruct[p,s]

			framesUnselectedperVideo = np.copy(nFramesperVideoperStruct[:,s])
			framesUnselectedperVideo[patientidperVideo != p] = 0

			while framesToSelect > 0 :

				# Number of videos with frames left
				nonzerocounts = (framesUnselectedperVideo != 0).sum()
				if nonzerocounts == 0 :
					break

				if (framesToSelect >= nonzerocounts) :
					framesperVideo = framesToSelect // nonzerocounts

					for vd in range(0,nVids) :
						if (framesUnselectedperVideo[vd] - framesperVideo >= 0) :
							framesUnselectedperVideo[vd] -= framesperVideo
							framesToSelect -= framesperVideo
							framesSelectedperVideoperStruct[vd,s] += framesperVideo
						else :
							framesToSelect -= framesUnselectedperVideo[vd]
							framesSelectedperVideoperStruct[vd,s] += framesUnselectedperVideo[vd]
							framesUnselectedperVideo[vd] = 0

				# Select randomly from patientsvidname = filename.rsplit('/',1)[1].rsplit('.',1)[0] + '.avi'
				else :
					nonzeroVideos = [vd for vd,val in enumerate(framesUnselectedperVideo) if val > 0 ]
					chosenVideos = np.random.choice(nonzeroVideos,framesToSelect,False)

					for vd in chosenVideos :
						framesSelectedperVideoperStruct[vd,s] += 1
						framesToSelect -= 1

	# Open an output file
	with open(outputfilename,'w') as outfile :
		outfile.write('filename frameno radius centrey centrex label ori phase' + os.linesep)
		outfile.write("background " + " ".join(substructurenames) + os.linesep)
		outfile.write(str(2*framesSelectedperVideoperStruct.sum()) + os.linesep)

		# Loop through each video and select random frames to go into the dataset
		# Loop through trackfiles
		for vd,filename in enumerate(trackfiles_ex):
			# Name of the video
			vidname = os.path.basename(os.path.splitext(filename)[0]) + '.avi'

			# Name of the corresponding heart track file
			heart_filename = os.path.join( heart_track_directory , os.path.basename(os.path.splitext(filename)[0]) + '.tk')
			heart_table,image_dims,_,heart_radius = ut.readHeartTrackFile(heart_filename)

			# The radius at which to perform detection is a specified fraction of the
			# heart radius
			radius = heart_radius*radius_frac

			# Try and read the mask for this video
			mask_string = os.path.join( mask_directory , vidname.rsplit('_',1)[0] + '_mask.png')
			mask = cv2.imread(mask_string,0)
			if mask is None :
				print 'Could not open mask file', mask_string
			edgedistimage = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

			# This list will contain all the frames chosen over all the structures
			all_chosen_frames = []

			# This list will contain all the info (in string form) to print for each line of the
			# chosen frames list
			all_frame_info = []

			for s,struct in enumerate(substructurenames) :

				# Read in the track data for this video
				table = ut.readSubstructure(filename,struct)

				# Cycle through views and choose a random selection of frames for each view
				# To be availabe a frame must have the structure present and have a frame number greater than "first frame"
				framesAvailable = np.array([ int(table[i,ut.tk_frameCol]) for i in range(0,len(table))
											if (int(table[i,ut.tk_presentCol]) == 1)
												and (int(table[i,ut.tk_frameCol]) >= first_frame)
												and (edgedistimage[int(table[i,ut.tk_yposCol]),int(table[i,ut.tk_xposCol])] - 2 > radius*overlap_frac)
												and (int(heart_table[i,ut.tk_viewCol]) in viewsPerStructure[s])
											],int)

				if len (framesAvailable) > 0 :
					# Choose some random frames from this view to add to the list
					framesChosen = np.array(np.random.choice(framesAvailable,framesSelectedperVideoperStruct[vd,s],False),int)
				else :
					framesChosen = []

				for f in framesChosen :

					xpos = table[f,ut.tk_xposCol]
					ypos = table[f,ut.tk_yposCol]

					# Write this positive example to file
					frameinfo = vidname + ' '
					frameinfo += str(f) + ' '
					if list_heart_radius :
						frameinfo += str(heart_radius) + ' '
					else :
						frameinfo += str(radius) + ' '
					frameinfo += str(int(ypos)) + ' '
					frameinfo += str(int(xpos)) + ' '
					frameinfo += str(int(s)+1) + ' '
					frameinfo += str(int(table[f,ut.tk_oriCol])) + ' '
					frameinfo += "0.0" + os.linesep

					all_chosen_frames += [f]
					all_frame_info += [frameinfo]

					# Select a random negative example from this frame that is within the permissible area
					while True :
						bgxpos = np.random.randint(int(math.ceil(radius))+1,int(math.floor(image_dims[0]-radius))-1)
						bgypos = np.random.randint(int(math.ceil(radius))+1,int(math.floor(image_dims[1]-radius))-1)

						struct_dist = ((bgxpos-xpos)**2 + (bgypos-ypos)**2)**0.5

						edgedist = edgedistimage[bgypos,bgxpos]

						if struct_dist/radius > 0.3 and edgedist - 2 > radius*overlap_frac :
							break

					# Generate a random orientation for this background example
					bgOri = np.random.randint(0,359)

					# Write this background example to file
					frameinfo = vidname + ' '
					frameinfo += str(f) + ' '
					if list_heart_radius :
						frameinfo += str(heart_radius) + ' '
					else :
						frameinfo += str(radius) + ' '
					frameinfo += str(int(bgypos)) + ' '
					frameinfo += str(int(bgxpos)) + ' '
					frameinfo += '0 '
					frameinfo += str(int(bgOri)) + ' '
					frameinfo += '0.0'  + os.linesep

					all_chosen_frames += [f]
					all_frame_info += [frameinfo]

			# Sort the information for this video
			to_write = [info for (frame,info) in sorted(zip(all_chosen_frames,all_frame_info), key=lambda pair: pair[0])]
			outfile.writelines(to_write)


# Execute as a script by parsing command line arguments and calling the above function
if __name__ == "__main__" :

	# Parse command line arguements
	parser = ap.ArgumentParser(description='Create a dataset from a set of trackfiles')
	parser.add_argument('subs_track_directory',help="directory containing the substructure track files")
	parser.add_argument('heart_track_directory',help="directory containing the heart track files")
	parser.add_argument('radius_frac',type=float,help="fraction of the heart radius at which to detect structures")
	parser.add_argument('substructures_file',help="name of the file containing substructures to use")
	parser.add_argument('--view_label','-v',type=int,default=-1,help="Only include substructures from this view label, ignore others")
	parser.add_argument('--mask_directory','-m',type=str,help="directory containing the image masks",default='')
	parser.add_argument('--n_examples_per_struct','-n',type=int,help="number of examples per substructure",default=100)
	parser.add_argument('--outfilename','-o',help="name of the output file",default="dataset")
	parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
	parser.add_argument('--first_frame','-f',type=int,help="avoid frames before this frame",default=0)
	parser.add_argument('--overlap_frac','-l',type=float,help="fraction of the radius that must be within the masked area for a background sample",default=1.0)
	parser.add_argument('--list_heart_radius','-r',help="list the heart radius rather than the substructure radius in the dataset file",action="store_true")
	args = parser.parse_args()

	# Call the function
	makeSubsDataset(args.subs_track_directory,args.heart_track_directory,args.radius_frac,args.substructures_file,args.view_label,args.mask_directory,args.n_examples_per_struct,args.outfilename,args.exclude_list,args.first_frame,args.overlap_frac,args.list_heart_radius)
