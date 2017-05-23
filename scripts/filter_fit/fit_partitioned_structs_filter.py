#!/usr/bin/python

# Imports
from os import linesep
import sys
import numpy as np
import CPBUtils as ut
import argparse as ap

def fit_partitioned_structs_filter(subs_track_directory, heart_track_directory, substructures_list_file, outfilename='filterout', windowlength=1, nviews=3, weight_precision=1.0, exclude_list=[], mean_shift_params=None, hidden_equilibrium_fraction=np.nan, hidden_time_constant=np.nan, hidden_weight=np.nan):

	# Find track files
	error_flag,trackfiles_ex = ut.getTracksList(subs_track_directory,'.stk',exclude_list)

	if error_flag :
		exit()

	# Read in list of substructures from file
	structure_names,viewsPerStructure,structuresPerView,systole_only,fourier_orders_per_struct = ut.readSubstructureList(substructures_list_file)
	n_structures = len(structure_names)
	n_structure_pairs = n_structures*(n_structures-1)/2

	# Open the output file for writing
	with open(outfilename,'w') as outfile:

		# First print the contents of a mean-shift parameters file if there is one provided
		if mean_shift_params is not None:
			with open(mean_shift_params,'r') as infile :
				lines = infile.readlines()
			if not np.isnan(hidden_equilibrium_fraction):
				lines[4] = str(hidden_equilibrium_fraction) + linesep
			if not np.isnan(hidden_time_constant):
				lines[7] = str(hidden_time_constant) + linesep
			if not np.isnan(hidden_weight):
				lines[10] = str(hidden_weight) + linesep
			outfile.writelines(lines)
			outfile.write(linesep)

		outfile.write("# Number of views" + linesep + str(nviews) + 2*linesep)
		outfile.write("# Total number of structures" + linesep + str(n_structures) + 2*linesep)
		outfile.write("# Structure Names and Views" + linesep)
		for name,views in zip(structure_names, viewsPerStructure) :
			outfile.write(name + " " + " ".join(str(v) for v in views) + linesep)
		outfile.write(linesep)

		# Loop over the different views
		for v in range(1,nviews+1) :

			n_structures_this_view = len(structuresPerView[v])
			n_structure_pairs_this_view = n_structures_this_view*(n_structures_this_view-1)/2
			fourier_orders_this_view = [fourier_orders_per_struct[s] for s in structuresPerView[v]]
			Dmax = max(fourier_orders_this_view)

			# List to store parameters
			list_params = [np.array([]).reshape(0,2*(2*fourier_order+1)) for fourier_order in fourier_orders_this_view]

			# Loop over trackfiles gathering the relevant information
			for vid,filename in enumerate(trackfiles_ex) :

				# Read the heart track file to get the heart radius
				heart_filename = os.path.join( heart_track_directory , os.path.basename(os.path.splitext(filename)[0]) + '.tk')
				heart_table,_,_,heart_radius = ut.readHeartTrackFile(heart_filename)

				# Read in the track data for this video for the substructures in this view
				substructure_tables = [ ut.readSubstructure(filename,structure_names[s]) for s in structuresPerView[v] ]

				# Split video into cycles to fit individual models to
				section_starts,section_ends = ut.findHeartCycleSections(heart_table,v,windowlength)

				# Loop over sections
				for sec in range(0,len(section_starts)) :

					n_data = section_ends[sec] - section_starts[sec]

					# Don't try and fit to a very short section
					if n_data < Dmax :
						continue

					# Loop over all structures for this view
					for s,s_table in enumerate(substructure_tables) :

						# Dimension of the phase prediction problem
						fourier_order = fourier_orders_this_view[s]
						D = 2*fourier_order+1

						# Fit a Fourier basis model for this substructure in this video section
						fourier_model = ut.fitStructureFourierModel( s_table[section_starts[sec]:section_ends[sec],:] , heart_table[section_starts[sec]:section_ends[sec],:] , heart_radius , fourier_order , weight_precision )

						# If 'None' is returned, then the structure was not present in these frames
						if fourier_model is None :
							# Put nans in the model for now, we'll fix this later
							list_params[s] = np.vstack([list_params[s],np.nan * np.ones(2*D) ])
						else :
							list_params[s] = np.vstack([list_params[s],fourier_model.flatten().T])

			# Score matrix to contain the entropy of the distribution of each substructure
			# relative to the heart (i.e. the absolute distribution of the relative parameters)
			abs_scores = np.zeros( n_structures_this_view )

			outfile.write('# Start View'+linesep+str(v)+linesep+'# Number of structures in this view'+linesep+str(n_structures_this_view)+2*linesep)
			for s_t in range(n_structures_this_view) :

				# Remove any row containing a nan
				nanless = list_params[s_t][~np.isnan(list_params[s_t]).any(axis=1)]

				# The mean parameter vector
				mean_param = np.mean(nanless,axis=0)

				# The covariance here is simply the covariance of the parameters
				covar_param = np.cov(nanless.T)

				# The entropy score is a monotonic function of determinant of the covariance matrix
				abs_scores[s_t] = np.linalg.slogdet(covar_param)[1]

				# Write to the file
				outfile.write("# Structure name" + linesep + structure_names[structuresPerView[v][s_t]] + linesep)
				outfile.write("# Fourier expansion order" + linesep + str(fourier_orders_this_view[s_t]) + linesep)
				outfile.write("# Systole Only" + linesep + ('1' if systole_only[structuresPerView[v][s_t]] else '0' ) + linesep)
				outfile.write("# Mean" + linesep)
				mean_param.tofile(outfile,sep=' ')
				outfile.write(linesep)
				outfile.write("# Covariance" + linesep)
				for row in covar_param:
					row.tofile(outfile,sep=" ")
					outfile.write(linesep)
				outfile.write(linesep)

			# **** NB ****
			# Everything from here on is currently pointless - will find the optimal
			# tree structure but seems to be very little improvement over star structure
			# Added this comment as it doesn't even work now that the structures can have a
			# different Fourier order
			if False:
				# Score matrix to contain the entropy of the conditional
				# distribution of each substructure to each other
				rel_scores = np.zeros( [ n_structures_this_view ]*2  )

				for s_t in range(n_structures_this_view) : # target
					for s_o in range(n_structures_this_view) : # origin

						if s_t == s_o :
							rel_scores[s_t,s_o] = 1e10 # some big number so this won't be chosen
							continue

						# Need to remove rows where there are any nans
						target_valid = np.logical_not(np.isnan(list_params[s_t]).any(axis=1))
						origin_valid = np.logical_not(np.isnan(list_params[s_o]).any(axis=1))
						valid_list = np.logical_and(target_valid,origin_valid)

						targets = list_params[s_t][valid_list,:]
						origins = list_params[s_o][valid_list,:]

						offset = targets - origins

						# Covariance of the residual
						covar = np.cov(offset.T)

						# The entropy score is a monotonic function of determinant of the covariance matrix
						rel_scores[s_t,s_o] = np.linalg.slogdet(covar)[1]

				# A list of structures that need to be located
				to_locate_list = range(n_structures_this_view)

				# A list of structures already located
				located_list = []

				# Now we need to loop through and iteratively locate the structures in this view relative to
				# those already localised
				for it in range(n_structures_this_view) :

					# The best we can do with absolute localisation
					best_abs_target = to_locate_list[np.argmin(abs_scores[to_locate_list])]
					best_abs_score = abs_scores[best_abs_target]

					# Now try with locations of previous targets known
					if located_list :
						reduced_rel_scores = rel_scores[np.ix_(to_locate_list,located_list)]
						best_rel_target, best_origin = np.unravel_index(np.argmin(reduced_rel_scores),reduced_rel_scores.shape)
						best_rel_target = to_locate_list[best_rel_target]
						best_origin = located_list[best_origin]
						best_rel_score = rel_scores[best_rel_target,best_origin]
					else :
						best_rel_score = 1e10

					if best_rel_score < best_abs_score :
						best_target = best_rel_target
						print "Target", structure_names[structuresPerView[v][best_target]], "Origin", structure_names[structuresPerView[v][best_origin]]
					else :
						best_target = best_abs_target
						print "Abs target", structure_names[structuresPerView[v][best_target]]


					# Move the chosen target from the to_locate_list to the located_list
					to_locate_list.remove(best_target)
					located_list.append(best_target)

# Setup argument parser and parse arguments
if __name__ == '__main__' :

	parser = ap.ArgumentParser(description='Create a substructure filter definition file for a sequential partitioned particle filter using Fourier series motion description')
	parser.add_argument('subs_track_directory',help="directory containing the substructure track files")
	parser.add_argument('heart_track_directory',help="directory containing the heart track files")
	parser.add_argument('substructures_list_file',help="name of the substructure list file to use")
	parser.add_argument('--outfilename','-o',help="name of the output file",default="filterout")
	parser.add_argument('--windowlength','-W',type=int,help="length of one window in cycles",default=1)
	parser.add_argument('--nviews','-v',type=int,help="number of viewing planes",default=3)
	parser.add_argument('--weight_precision','-r',type=float,help="weight precision (inverse variance) to use as a regularising term", default=1.0)
	parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
	parser.add_argument('--mean_shift_params','-m',help="file containing mean shift parameters to copy and place at the start of the output model file")
	parser.add_argument('--hidden_equilibrium_fraction','-H',type=float,help='override the hidden equilibrium fraction parameter with this value',default=np.nan)
	parser.add_argument('--hidden_time_constant','-t',type=float,help='override the hidden time constant parameter with this value',default=np.nan)
	parser.add_argument('--hidden_weight','-w',type=float,help='override the hidden weight parameter with this value',default=np.nan)
	args = parser.parse_args()

	fit_partitioned_structs_filter(args.subs_track_directory, args.heart_track_directory, args.substructures_list_file, outfilename=args.outfilename, windowlength=args.windowlength, nviews=args.nviews, weight_precision=args.weight_precision, exclude_list=args.exclude_list, mean_shift_params=args.mean_shift_params, hidden_equilibrium_fraction=args.hidden_equilibrium_fraction, hidden_time_constant=args.hidden_time_constant, hidden_weight=args.hidden_weight)
