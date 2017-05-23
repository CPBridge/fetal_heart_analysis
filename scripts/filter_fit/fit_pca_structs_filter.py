#!/usr/bin/python

# Imports
from os import linesep
import sys             # argv
import numpy as np     # array, loadtxt
import math            # ceil
import argparse as ap  # parser (for arguments)
import CPBUtils as ut
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

def fit_pca_structs_filter(subs_track_directory, heart_track_directory, substructures_list_file, outfilename='filterout', windowlength=1, nviews=3, pcacomponents=5, animate=False, still_animations=False, weight_precision=1.0, exclude_list=[], mean_shift_params=None, hidden_equilibrium_fraction=np.nan, hidden_time_constant=np.nan, hidden_weight=np.nan):

	# Find track files
	error_flag,trackfiles_ex = ut.getTracksList(subs_track_directory,'.stk',exclude_list)

	if error_flag :
		exit()

	# Find the unique patients
	patients = [os.path.basename(f.rsplit('_',1)[0]) for f in trackfiles_ex]
	uniquePatients = list(set(patients))
	nPatients = len(uniquePatients)
	nVids = len(trackfiles_ex)

	# Read in list of substructures from file
	structure_names,viewsPerStructure,structuresPerView,systole_only,fourier_order_list = ut.readSubstructureList(substructures_list_file)
	n_structures = len(structure_names)

	# The dimension of the phase prediction dataset
	max_fourier_order = [0] + [max([fourier_order_list[i] for i in structuresPerView[v]]) for v in range(1,nviews+1)]
	Dmax = [2*x+1 for x in max_fourier_order]

	def get_position_array(i,temporal_model_matrix,v,n_frames):
		phi = i*2.0*np.pi/n_frames

		# Create the phase vector
		phi_vector = np.zeros([Dmax[v],1])
		phi_vector[0,0] = 1.0
		for n in range(1,max_fourier_order[v]+1) :
			phi_vector[2*(n-1)+1,0] = np.sin(n*phi)
			phi_vector[2*(n-1)+2,0] = np.cos(n*phi)

		position_array = np.matrix(temporal_model_matrix.T)*phi_vector
		position_array = position_array.reshape([-1,2])
		return position_array

	# Set up the animation function for later
	def animate_model(temporal_model_matrix,v) :
		# Display this model
		n_frames = 100
		fig,ax = plt.subplots()
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		points, = ax.plot([],[],'bo',ms=6)

		def init():
			points.set_data([],[])
			return points

		def step(i) :
			position_array = get_position_array(i,temporal_model_matrix,v,n_frames)
			points.set_data(position_array[:,0],position_array[:,1])
			return points

		ani = animation.FuncAnimation(fig,step,interval = 1,frames = n_frames, init_func = init, repeat = False)
		plt.show()


	def plot_still(temporal_model_matrix,v,annotate=False):
		n_frames = 20
		fig,ax = plt.subplots(figsize=(4,4))
		plt.xlim(-1.8,1.8)
		plt.ylim(-1.8,1.8)

		for i in range(n_frames):
			position_array = get_position_array(i,temporal_model_matrix,v,n_frames)
			plt.scatter(position_array[:,0],position_array[:,1],marker='x',color=cm.jet(float(i)/n_frames))
			if i == 0 and annotate:
				for s,x,y in zip(structuresPerView[v],position_array[:,0],position_array[:,1]):
					plt.annotate(structure_names[s],
					xy=(x, y), xytext=(2, 2),
					textcoords='offset points', ha='left', va='bottom',fontsize='x-small')

		fig.patch.set_facecolor('white')
		plt.show()

	# Lists of all temporal models stacked horizontally for each view
	all_temporal_models = [np.array([]).reshape(0,2*Dmax[v]*len(structuresPerView[v])) for v in range(1,nviews+1)]

	# Loop through trackfiles gathering the relevant information
	for vd,filename in enumerate(trackfiles_ex):

		# Read the heart track file to get the heart radius
		heart_filename = os.path.join( heart_track_directory , os.path.basename(os.path.splitext(filename)[0]) + '.tk')
		heart_table,_,_,heart_radius = ut.readHeartTrackFile(heart_filename)

		# This list will contain all the tracks for the substructures in this video
		substructure_tables = []
		# Read in the track info for the substructures
		for s,struct in enumerate(structure_names):
			# Read in the track data for this video
			substructure_tables += [ ut.readSubstructure(filename,struct) ]

		# Loop over the different views
		for v in range(1,nviews+1) :

			# Split video into cycles to fit individual models to
			section_starts,section_ends = ut.findHeartCycleSections(heart_table,v,window_length=windowlength)

			# Loop over sections
			for sec_start,sec_end in zip(section_starts,section_ends) :

				n_data = sec_end - sec_start

				# Don't try and fit to a very short section
				if n_data < Dmax[v] :
					continue

				temporal_model_one_section = np.zeros([Dmax[v],2*len(structuresPerView[v])])

				# Loop over all structures for this view
				for subind,sub in enumerate(structuresPerView[v]) :

					# Fit a Fourier basis model for this substructure in this video section
					fourier_model = ut.fitStructureFourierModel( substructure_tables[sub][sec_start:sec_end,:] , heart_table[sec_start:sec_end,:] , heart_radius , fourier_order_list[sub] , weight_precision=weight_precision, output_order=max_fourier_order[v])

					# If 'None' is returned, then the structure was not present in these frames
					if fourier_model is None :
						# Put nans in the model for now, we'll fix this later
						temporal_model_one_section[:,2*subind:2*(subind+1)] = np.nan
					else :
						temporal_model_one_section[:,2*subind:2*(subind+1)] = fourier_model

				# Add this to the list of overall temporal models
				all_temporal_models[v-1] = np.vstack([all_temporal_models[v-1],temporal_model_one_section.flatten()])

				if animate > 1:
					# Replace the nans with zeros for the purpose of animation
					nanless_model = temporal_model_one_section
					nanless_model[np.isnan(nanless_model)] = 0.0
					animate_model(nanless_model,v)

	with open(outfilename,'w') as outfile :

		# Copy mean shift paramters if required
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
		outfile.write("# Reduced state dimension" + linesep + str(pcacomponents) + 2*linesep)
		outfile.write("# Total number of structures" + linesep + str(n_structures) + 2*linesep)
		outfile.write("# Structure Names and Views" + linesep)
		for name,views,so in zip(structure_names, viewsPerStructure, systole_only) :
			outfile.write(name + " " + ('1' if so else '0') + " " + " ".join(str(v) for v in views) + linesep)
		outfile.write(linesep)

		# Loop over the different views
		for v in range(1,nviews+1) :

			n_samples = all_temporal_models[v-1].shape[0]

			# Find the mean model and subtract to give centred data
			# Ignore nans when finding mean using nanmean
			mean_temporal_model = np.nanmean(all_temporal_models[v-1],axis=0)
			centred = all_temporal_models[v-1] - mean_temporal_model[np.newaxis,:]

			# Replace nans with zeros (which after centering now represent the mean)
			centred[np.isnan(centred)] = 0.0

			# Animate the average model
			if animate > 0:
				print "View", v, "mean"
				if still_animations:
					plot_still(mean_temporal_model.reshape([Dmax[v],2*len(structuresPerView[v])]),v,annotate=True)
				else:
					animate_model(mean_temporal_model.reshape([Dmax[v],2*len(structuresPerView[v])]),v)

			# Find the SVD for principal component analysis
			_,S,V = np.linalg.svd(centred)

			# The standard deviations of the principal component coefficients are
			# related to the singular values
			principal_std_devs = (S[:pcacomponents]**2/(n_samples-1))**0.5

			# Take only the top few principal axes and scale them by the standard
			# deviations so that the coefficients have unit standard deviation
			Vreduced = np.matrix(V[:pcacomponents,:] * principal_std_devs[:,np.newaxis])

			# Output to the file
			outfile.write("# View " + str(v) + " max Fourier expansion order" + linesep + str(max_fourier_order[v]) + 2*linesep)
			outfile.write("# View " + str(v) + " mean")
			outfile.write(linesep)
			mean_temporal_model.tofile(outfile,sep=" ")
			outfile.write(2*linesep)
			outfile.write("# View " + str(v) + " scaled principal axes")
			outfile.write(linesep)
			for row in Vreduced.T:
				row.tofile(outfile,sep=" ")
				outfile.write(linesep)
			outfile.write(linesep)

			# Loop through PCA components and animate them
			for p in range(0,pcacomponents) :
				vec = np.matrix(np.zeros([1,pcacomponents]))

				for mult in [-1.0,1.0] :
					vec[0,p] = mult*3.0 #*principal_std_devs[p]

					temporal_model_matrix = mean_temporal_model[np.newaxis,:] + vec*Vreduced
					if animate > 0:
						print "View", v, "component", p, mult*3.0, "standard deviations"
						if still_animations:
							plot_still(temporal_model_matrix.reshape([Dmax[v],2*len(structuresPerView[v])]),v)
						else:
							animate_model(temporal_model_matrix.reshape([Dmax[v],2*len(structuresPerView[v])]),v)

if __name__ == '__main__' :
	parser = ap.ArgumentParser(description='Create a filter definition file for substructures from a set of trackfiles')
	parser.add_argument('subs_track_directory',help="directory containing the substructure track files")
	parser.add_argument('heart_track_directory',help="directory containing the heart track files")
	parser.add_argument('substructures_list_file',help="name of the substructure list file to use")
	parser.add_argument('--outfilename','-o',help="name of the output file",default="filterout")
	parser.add_argument('--windowlength','-W',type=int,help="length of one window in cycles",default=1)
	parser.add_argument('--nviews','-v',type=int,help="number of viewing planes",default=3)
	parser.add_argument('--pcacomponents','-p',type=int,help="Number of PCA components",default=5)
	parser.add_argument('--animate','-a',type=int,help="animate each of the models as they are calculated, 1 animates the final models, 2 animates all intermediate results")
	parser.add_argument('--still_animations','-s',action='store_true',help='Produce a single still frame for each animation rather than a sequence')
	parser.add_argument('--weight_precision','-r',type=float,help="weight precision (inverse variance) to use as a regularising term", default=1.0)
	parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
	parser.add_argument('--mean_shift_params','-m',help="file containing mean shift parameters to copy and place at the start of the output model file")
	parser.add_argument('--hidden_equilibrium_fraction','-H',type=float,help='override the hidden equilibrium fraction parameter with this value',default=np.nan)
	parser.add_argument('--hidden_time_constant','-t',type=float,help='override the hidden time constant parameter with this value',default=np.nan)
	parser.add_argument('--hidden_weight','-w',type=float,help='override the hidden weight parameter with this value',default=np.nan)
	args = parser.parse_args()

	fit_pca_structs_filter(args.subs_track_directory, args.heart_track_directory, args.substructures_list_file, outfilename=args.outfilename, windowlength=args.windowlength, nviews=args.nviews, pcacomponents=args.pcacomponents, animate=args.animate, still_animations=args.still_animations, weight_precision=args.weight_precision, exclude_list=args.exclude_list, mean_shift_params=args.mean_shift_params, hidden_equilibrium_fraction=args.hidden_equilibrium_fraction, hidden_time_constant=args.hidden_time_constant, hidden_weight=args.hidden_weight)
