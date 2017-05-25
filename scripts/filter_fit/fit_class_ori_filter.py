#!/usr/bin/python

# Imports
from os import linesep
import sys             # argv
import numpy as np     # array, loadtxt
import scipy.stats     # circmean, circvar
import argparse as ap  # parser (for arguments)
import glob            # glob
import math 		   # hypot
import CPBUtils as ut

def fit_class_ori_filter(track_directory,outfilename,n_views,mean_shift_params='',excludelist=[],hidden_equilibrium_fraction=np.nan,hidden_time_constant=np.nan,hidden_weight=np.nan):

	# Get a list of trackfiles
	error_flag,trackfiles_ex = ut.getTracksList(track_directory,'.tk',excludelist)
	if error_flag:
		exit()

	# Initialise containers for results
	array_trans_counts = np.eye(n_views,dtype=int)
	array_norm_spat_abs_offset = np.zeros([n_views,n_views],float)
	array_norm_spat_abs_offset_squared = np.zeros([n_views,n_views],float)

	# 2D list, where each element is an empty numpy array
	list_2D_offsets = [[np.matrix([]).reshape(0,2)]*n_views for _ in range(n_views)]
	list_ang_offsets = [[np.array([])]*n_views for _ in range(n_views)]

	# Loop through trackfiles
	for vd,filename in enumerate(trackfiles_ex):

		# Find the radius in the header information
		with open(filename, 'r') as openfile:
			openfile.readline() # header line
			openfile.readline() # dimensions lineabs_
			radius = float(openfile.readline().split()[1]) # headup and radius line

		# Read inthe rest of the data as a table
		table = np.loadtxt(filename, skiprows=3)
		nFrames = int(table[-1][ut.tk_frameCol]) + 1

		# Loop through the frames looking for transitions
		f = 0
		while f < nFrames -1:

			# Skip frames where there is no label, or the heart is not visible
			if int(table[f][ut.tk_labelledCol]) == 0 or int(table[f][ut.tk_presentCol]) == 0  :
				f += 1
				continue

			thisview = int(table[f][ut.tk_viewCol])

			# There is no transition if the next frame is of the same class
			if int(table[f+1][ut.tk_presentCol]) == 1 and int(table[f+1][ut.tk_viewCol]) == thisview :
				f += 1
				continue

			# If we get here there is some sort of transition, though it may be to the same class
			# Loop to find the next frame with a labelled view
			ff = f + 1
			while ff < nFrames :
				if int(table[ff][ut.tk_presentCol]) == 1  and int(table[f][ut.tk_labelledCol]) == 1:
					break
				ff += 1

			# Check that the end of the video was not reached
			if ff == nFrames:
				break

			# Check for a transition back to the same view
			nextview = int(table[ff][ut.tk_viewCol])
			if(nextview == thisview):
				f = ff
				continue

			# If we get here there is definitely a transition betweeen two different classes

			# Define the direction of the offset to be moving from the lower class to the higher class
			if thisview < nextview:
				frame1 = f
				frame2 = ff
				c1 = thisview
				c2 = nextview
			else :
				frame1 = ff
				frame2 = f
				c1 = nextview
				c2 = thisview

			x1 = float(table[frame1][ut.tk_xposCol])
			y1 = float(table[frame1][ut.tk_yposCol])
			ori1 = float(table[frame1][ut.tk_oriCol])*np.pi/180.0

			x2 = float(table[frame2][ut.tk_xposCol])
			y2 = float(table[frame2][ut.tk_yposCol])
			ori2 = float(table[frame2][ut.tk_oriCol])*np.pi/180.0

			# Update the count
			array_trans_counts[c1-1][c2-1] += 1
			array_trans_counts[c2-1][c1-1] += 1

			# The spatial offset distance between the two view, normalised by radius
			norm_spat_abs_offset = math.hypot(x2-x1,y2-y1)/radius
			array_norm_spat_abs_offset[c1-1][c2-1] += norm_spat_abs_offset
			array_norm_spat_abs_offset[c2-1][c1-1] += norm_spat_abs_offset
			array_norm_spat_abs_offset_squared[c1-1][c2-1] += norm_spat_abs_offset**2
			array_norm_spat_abs_offset_squared[c2-1][c1-1] += norm_spat_abs_offset**2

			# The 2D offset - work in (x,y) coordinates with x defined with left positive and y defined to be up positive (NB this means y is swapped from trackfile definition)
			norm_offset_2D = np.matrix([x2-x1,y1-y2]).T/radius
			rotmat12 = np.matrix([[np.cos(-ori1),-np.sin(-ori1)],[np.sin(-ori1), np.cos(-ori1)]])
			norm_offset_2D_rel1 = rotmat12*norm_offset_2D
			rotmat21 = np.matrix([[np.cos(-ori2),-np.sin(-ori2)],[np.sin(-ori2), np.cos(-ori2)]])
			norm_offset_2D_rel2 = -rotmat21*norm_offset_2D

			# Store these in the arrays
			list_2D_offsets[c1-1][c2-1] = np.vstack([list_2D_offsets[c1-1][c2-1],norm_offset_2D_rel1.T])
			list_2D_offsets[c2-1][c1-1] = np.vstack([list_2D_offsets[c2-1][c1-1],norm_offset_2D_rel2.T])

			# The orientation of the second class relative to the first
			# Wrapped to -pi to pi
			ori_offset = ori2 - ori1
			list_ang_offsets[c1-1][c2-1] = np.append(list_ang_offsets[c1-1][c2-1],ori_offset)
			list_ang_offsets[c2-1][c1-1] = np.append(list_ang_offsets[c2-1][c1-1],-ori_offset)

			# Start seraching again rom the other end of the transition
			f = ff

			#print "frame1", frame1, "frame2", frame2, "v1", thisview, "v2", nextview, "x1", x1, "y1", y1, "ori1", ori1, "x2", x2, "y2", y2, "ori2", ori2, "norm_spat_offset", norm_spat_offset, "abs_spat_offset_ang", abs_spat_offset_ang, "rel_spat_offset_ang", rel_spat_offset_ang, "ori_offset", ori_offset


	# Calculate mean and standard deviation
	norm_spat_abs_offset_mean = array_norm_spat_abs_offset/array_trans_counts
	norm_spat_abs_offset_std = np.sqrt((array_norm_spat_abs_offset_squared - (array_norm_spat_abs_offset**2)/(array_trans_counts))/(array_trans_counts+1))

	with open(outfilename,'w') as outfile :

		#outfile.write("Mean normalised transition offset" + linesep)
		#norm_spat_abs_offset_mean.tofile(outfile,sep=" ")
		#outfile.write(linesep + "Standard deviation of normalised transition offset" + linesep)
		#norm_spat_abs_offset_std.tofile(outfile,sep=" ")
		outfile.write("# Number of view classes (excluding background)" + linesep + str(n_views))

		for c1,row in enumerate(list_2D_offsets) :
			for c2,data in enumerate(row) :
				if(c1 != c2) :

					offsetmean = np.mean(data,0)
					offsetcovar = np.cov(data.T)

					outfile.write(linesep + linesep + "# Class transition " + str(c1+1) + " -> " + str(c2+1) + " spatial (2D) offset mean" + linesep)
					offsetmean.tofile(outfile,sep=" ")
					outfile.write(linesep + linesep + "# Class transition " + str(c1+1) + " -> " + str(c2+1) + " spatial (2D) offset covariance" + linesep)
					offsetcovar.tofile(outfile, sep= " ")

		for c1,row in enumerate(list_ang_offsets) :
			for c2,data in enumerate(row) :
				if(c1 != c2) :

					orioffsetmean = scipy.stats.circmean(data)
					orioffsetstd = scipy.stats.circstd(data)

					outfile.write(linesep + linesep + "# Class transition " + str(c1+1) + " -> " + str(c2+1) + " angular offset mean" + linesep)
					orioffsetmean.tofile(outfile,sep=" ")
					outfile.write(linesep + linesep + "# Class transition " + str(c1+1) + " -> " + str(c2+1) + " angular offset standard deviation" + linesep)
					orioffsetstd.tofile(outfile, sep= " ")

		outfile.write(linesep + linesep)

		# Append a mean-shift parameters file if there is one provided
		if len(mean_shift_params) > 0:
			with open(mean_shift_params,'r') as infile :
				lines = infile.readlines()
			if not np.isnan(hidden_equilibrium_fraction):
				lines[6] = str(hidden_equilibrium_fraction) + linesep
			if not np.isnan(hidden_time_constant):
				lines[9] = str(hidden_time_constant) + linesep
			if not np.isnan(hidden_weight):
				lines[12] = str(hidden_weight) + linesep
			outfile.writelines(lines)

# Run this when the script is executed directly
if __name__ == '__main__':

	# Set up argument parser
	parser = ap.ArgumentParser(description='Work out spatial offsets for class transitions from track files')
	parser.add_argument('n_views',type=int,help="number of views types labelled (ex background)")
	parser.add_argument('track_directory',help="directory containing the track files")
	parser.add_argument('mean_shift_params',help="Name of a mean-shift parameters file to append to the offset parameters")
	parser.add_argument('outfilename',help="name of the output file")
	parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
	parser.add_argument('--cross_val','-c',action='store_true',help='Create filter files for each fold in a leave-one-out cross-validation')
	parser.add_argument('--hidden_equilibrium_fraction','-H',type=float,help='override the hidden equilibrium fraction parameter with this value',default=np.nan)
	parser.add_argument('--hidden_time_constant','-t',type=float,help='override the hidden time constant parameter with this value',default=np.nan)
	parser.add_argument('--hidden_weight','-w',type=float,help='override the hidden weight parameter with this value',default=np.nan)

	# Capture arguments to local variables for convenience
	args = parser.parse_args()

	if args.cross_val and args.exclude_list:
		print("ERROR: Cannot use cross_val option and provide an exclude list")
		sys.exit()

	# Make the function call
	if args.cross_val:
		patients_list = ut.getPatientsInTrackDirectory(args.track_directory)
		for patient in patients_list:
			fit_class_ori_filter(args.track_directory,args.outfilename+'_ex'+patient,args.n_views,mean_shift_params=args.mean_shift_params,excludelist=[patient],
				hidden_equilibrium_fraction=args.hidden_equilibrium_fraction,hidden_time_constant=args.hidden_time_constant,hidden_weight=args.hidden_weight)

	else:
		fit_class_ori_filter(args.track_directory,args.outfilename,args.n_views,mean_shift_params=args.mean_shift_params,excludelist=args.exclude_list,
	              hidden_equilibrium_fraction=args.hidden_equilibrium_fraction,hidden_time_constant=args.hidden_time_constant,hidden_weight=args.hidden_weight)
