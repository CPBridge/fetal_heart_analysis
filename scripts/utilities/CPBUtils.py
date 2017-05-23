import numpy as np
import math
import glob
import os

# Constants for (heart) trackfile definition
tk_frameCol = 0
tk_labelledCol = 1
tk_presentCol = 2
tk_yposCol = 3
tk_xposCol = 4
tk_oriCol = 5
tk_viewCol = 6
tk_phasePointsCol = 7
tk_cardiacPhaseCol= 8

# Constants for substructure track file definition
stk_frameCol = 0
stk_labelledCol = 1
stk_presentCol = 2
stk_yposCol = 3
stk_xposCol = 4
stk_oriCol = 5

# Constants for abdomen trackfile definition
atk_frameCol = 0
atk_labelledCol = 1
atk_presentCol = 2
atk_yposCol = 3
atk_xposCol = 4
atk_oriCol = 5

# Constants for the dataset definition
ds_vidCol = 0
ds_frameCol = 1
ds_radiusCol = 2
ds_centreyCol = 3
ds_centrexCol = 4
ds_viewCol = 5
ds_oriCol = 6
ds_phaseCol = 7

# Constants for summary files
sum_col_model = 0
sum_col_video = 1
sum_col_frames = 2
sum_col_correct_detections = 3
sum_col_correct_rejections = 4
sum_col_misses = 5
sum_col_incorrect_detections = 6
sum_col_incorrect_locations = 7
sum_col_class_confusions = 8
sum_col_obscured_detections = 9
sum_col_obscured_rejections = 10
sum_col_obscured_incorrect_locations = 11
sum_col_obscured_class_confusions = 12
sum_col_confusion_matrix_first = 13
sum_col_confusion_matrix_last = 28
sum_col_obscured_confusion_matrix_first = 29
sum_col_obscured_confusion_matrix_last = 44
sum_col_ori_error = 45
sum_col_phase_error = 46
sum_col_time = 47
sum_col_struct_first = 48

# Constants for structures within the summary files
sum_num_cols_per_struct = 6
sum_struct_name_col_offset = 0
sum_struct_correct_detections_col_offset = 1
sum_struct_correct_rejections_col_offset = 2
sum_struct_misses_col_offset = 3
sum_struct_incorrect_detections_col_offset = 4
sum_struct_distance_col_offset = 5


def getTracksList(directory,ext,exclude_list=[]) :

	error = False

	# Check that the extension starts with a dot
	if not ext.startswith('.') :
		ext = '.' + ext

	# Get a list of all the matching files in the directory
	trackfiles = glob.glob(os.path.join(directory,'*'+ext))

	if len(trackfiles) == 0 :
		print "ERROR getting trackfiles, no matching files found in", directory
		error = True

	# Exclude all the listed patients
	trackfiles_ex = [tf for tf in trackfiles if os.path.basename(tf.rsplit('_',1)[0]) not in exclude_list]
	unused_excludes = [ex for ex in exclude_list if ex not in [os.path.basename(f.rsplit('_',1)[0]) for f in trackfiles]]

	if len(unused_excludes) > 0 :
		print "ERROR: The following excluded patients were not found: "
		for ex in unused_excludes :
			print ex
		error = True

	return error,trackfiles_ex



# Function to read a single substructure track from a trackfile
def readSubstructure(filename,substructure):
	'''
	Read data for a single specified structure from a structure track (.stk)
	file.

	Arguments:
		filename	--	String containing file path and name for the .stk file to read
						from
		structure	--	String containing name of structure to read

	Returns:
		table		--	numpy array containing data for the structure. Rows contain
						frames, columns contain variables.
						The columns may be indexed by the variables stk_frameCol, stk_labelledCol,
						stk_presentCol, stk_yposCol, stk_xposCol, and stk_oriCol.
						If the requested structure is not in the file, the return value will be
						None.
	'''
	# Read in text in one monlithic block
	with open(filename,'r') as infile :
		text = infile.read()

	# Split into blocks based on empty lines
	textblocks = text.split("\n\n")

	# Loop through the blocks
	for block in textblocks[1:] :
		blocklines = block.split("\n")
		if not blocklines[0] :
			continue
		structname = blocklines[0].split()[1]

		if structname == substructure :
			table = np.zeros([len(blocklines)-1,6],dtype=int)

			# We have found the right block, now need to extract the data
			for i,dataline in enumerate(blocklines[1:]) :
				table[i,:] = np.fromstring(dataline,sep=" ")

			# We have found the right block, now need to extract the data
			for tableline,dataline in zip(blocklines[1:],table) :
				tableline = np.fromstring(dataline,sep=" ")

			return table

	return None


def readSubstructureTrackAsDicts(filename,substructure):

	table = readSubstructure(filename,substructure)

	return [ { 'labelled' : row[stk_labelledCol] > 0,
	           'y'        : row[stk_yposCol    ],
	           'x'        : row[stk_xposCol    ],
	           'ori'      : row[stk_oriCol     ],
	           'present'  : row[stk_presentCol ],
			 } for row in table]


def readMultipleStructuresTrackAsDicts(filename,structure_list):

	list_of_tracks = [readSubstructureTrackAsDicts(filename,struct) for struct in structure_list]

	# This monstrosity transposes the above to give the list by frame (outer)
	# then structure (inner)
	transposed_list = list(map(list,zip(*list_of_tracks)))

	# Now a dictionary comprehension nested inside a list comprehension, FML
	# This is a list (by frame) of dictionaries (of structures) of dictionaries (of variables)
	return [ { struct : data for struct,data in zip(structure_list,frame_row)} for frame_row in transposed_list]


def readSubstructureList(filename) :
	names_list = []
	viewsPerStructure = []
	systole_only = []
	fourier_order = []

	# Open the file
	with open(filename,'r') as infile:
		for line in infile:
			names_list += [line.split()[0]]
			fourier_order += [int(line.split()[1])]
			systole_only += [line.split()[2] != '0']
			viewsPerStructure += [map(int,line.split()[3:])]

	# Largest view index used
	maxView = max(map(max,viewsPerStructure))

	# Find a list of the structures present in each view
	structuresPerView = [ [struct for struct in range(len(names_list)) if view in viewsPerStructure[struct]] for view in range(maxView+1)]

	return names_list, viewsPerStructure, structuresPerView, systole_only, fourier_order



# Reads in the information in a heart track file and returns
def readHeartTrackFile(filename) :
	'''
	Read data from a heart track (.tk) file.

	Arguments:
		filename -- String containing file path and name for the .tk file to read from

	Returns:
		table		--	numpy array containing the data for each frame.
						Each row corresponds to a frame and each column corresponds to a variable.
						The columns my be indexed by the variables tk_frameCol, tk_labelledCol,
						tk_presentCol, tk_yposCol, tk_xposCol, tk_oriCol, tk_viewCol,
						tk_phasePointsCol, tk_cardiacPhaseCol
		image_dims	--	2-element list containing width and height of the video in
						pixels
		headup		-- 	Boolean variable indicating the 'flip' of the video. True
						indicates that the cross section is being viewed along the direction from
						the fetal head to the fetal toes. False indicates the other direction.
	'''

	# Open the file to read info on the first couple of lines
	with open(filename) as infile :
		# Skip the first comment line
		infile.readline()

		# The next line contains the image dimensions
		image_dims = map(int,infile.readline().split())

		# The next line contains the headup and radius information
		line_info = infile.readline().split()
		headup = bool(line_info[0])
		radius = float(line_info[1])

	# Now use loadtxt to read the rest of the data
	table = np.loadtxt(filename,skiprows=3)
	return table,image_dims,headup,radius



# Reads the track file and return as rows of dictionaries
def readHeartTrackFileAsRows(filename) :
	'''
	Read data from a heart track (.tk) file and return the frame-wise
	annotations as a list of dictionaries.

	Arguments:
		filename	--	String containing file path and name for the .tk file to read from

	Returns:
		dict_rows	--	A list of dictionaries, with one dictionary per frame.
						For each dictionary, the keys 'frame', 'x', 'y', 'view', 'labelled',
						'present', 'ori' and 'phase' may be usedto access the relevant variables.
		image_dims	--	2-element list containing width and height of the video in
						pixels
		headup		--	Boolean variable indicating the 'flip' of the video. True
						indicates that the cross section is being viewed along the direction from
						the fetal head to the fetal toes. False indicates the other direction.
	'''
	# Use above function to get the data
	table,image_dims,headup,radius = readHeartTrackFile(filename)

	# Create list of independent empty dictionaries
	dict_rows = [{} for _ in range(len(table))]

	# Loop and add data to the dictionaries
	for dict_row,table_row in zip(dict_rows,table):
		dict_row['frame'] = int(table_row[tk_frameCol])
		dict_row['y'] = int(table_row[tk_yposCol])
		dict_row['x'] = int(table_row[tk_xposCol])
		dict_row['class'] = int(table_row[tk_viewCol])
		dict_row['labelled'] = int(table_row[tk_labelledCol]) > 0
		dict_row['present'] = int(table_row[tk_presentCol])
		dict_row['ori'] = float(table_row[tk_oriCol])*(math.pi/180.0)
		dict_row['phase'] = float(table_row[tk_cardiacPhaseCol])

	return dict_rows,image_dims,headup,radius


# Function that takes a heart table track and splits it temporally into
# sections where the required view is present. Each section contains approximately
# 'window_length' cardaic cycles. The returned lists repectively contain the
# indices of the start and end frames
def findHeartCycleSections(heart_table,view,window_length=1) :

	# The condition dictating whether a frame can be in a section
	def valid_condition(ff) :
		return heart_table[ff,tk_labelledCol] and heart_table[ff,tk_presentCol] > 0 and (heart_table[ff,tk_viewCol] == view)

	# Lists that will contain the start (inclusive) and end (exclusive)
	# frames of each section
	section_starts = []
	section_ends = []

	# Loop over all frames in the video
	in_section = False
	for f in range(len(heart_table)) :
		if in_section :
			if not valid_condition(f) :
				in_section = False
				section_ends += [f]
			elif heart_table[f][tk_phasePointsCol] in [1,2]:
				n_end_systole += 1
				if (n_end_systole >= window_length) :
					section_ends += [f]
					section_starts += [f]
					n_end_systole = 0
		else :
			if valid_condition(f) :
				in_section = True
				section_starts += [f]
				n_end_systole = 0

	if in_section :
		section_ends += [len(heart_table)]

	return section_starts,section_ends


# Takes a section of a substructure table and a matching section of a heart table
# and produces a linear-in-the-parameters Fourier model of the structure's motion
# over that interval. The resulting weights are returned as a numpy array of size Dx2
# with each column representing the weights for the x and y components.
# 'Order' is the Fourier order of the model being fitted for this substructure.
# If desired, the output order can be a higher order, in which case the rest of the
# weights will be set to zero at the output.
def fitStructureFourierModel(struct_table,heart_table,heart_radius,order,weight_precision=0.0,output_order=-1):

	# The data dimension
	D = 2*order+1

	# If the no value was provided for output order, assume that it's the same as order
	if output_order < 0 :
		output_order = order
	if output_order < order:
		raise AssertionError('Output order must be greater than or equal to order')
	output_D = 2*output_order+1

	phase_matrix = np.array([]).reshape(0,D)
	locations_matrix = np.array([]).reshape(0,2)

	n_frames = len(struct_table)
	if len(heart_table) != n_frames :
		raise AssertionError('The length of heart_table must be the same as the length of the structure table')

	# Loop over the frames in this video section
	for struct_row,heart_row in zip(struct_table,heart_table) :
		# Ignore this frame if the structure is not present
		if( struct_row[stk_labelledCol] and struct_row[stk_presentCol] > 0 ) :

			# Add a new row to the phase matrix and populate it with the relevant elements
			phi = float(heart_row[tk_cardiacPhaseCol])
			phase_matrix = np.vstack([phase_matrix,np.zeros([1,D])])
			phase_matrix[-1,0] = 1.0
			for n in range(1,order+1) :
				phase_matrix[-1,2*(n-1)+1] = np.sin(n*phi)
				phase_matrix[-1,2*(n-1)+2] = np.cos(n*phi)

			# Create a rotation matrix to move from frame to heart coordinates
			ori = float(heart_row[tk_oriCol])*np.pi/180.0
			rotmat = np.matrix([[np.cos(-ori),-np.sin(-ori)],[np.sin(-ori), np.cos(-ori)]])

			# Offset of this substructure from the heart centre (using x,y convention, up right positive)
			offset = np.array([[float(struct_row[stk_xposCol]),-float(struct_row[stk_yposCol])]]) - np.array([[float(heart_row[tk_xposCol]),-float(heart_row[tk_yposCol])]])
			# Normalise by radius
			offset /= heart_radius
			# Rotate into heart coordinate system
			offset = (rotmat*offset.T).T

			# Store this offset
			locations_matrix = np.vstack([locations_matrix,offset])

	# The model for this substructure and this section of the video in both x and y
	if locations_matrix.shape[0] > 0:
		# Add in regulariser term using a prior precision matrix
		# on the weights
		prior_matrix = (weight_precision**0.5)*np.eye(D)
		prior_matrix[0,0] = 0.0

		# Container for the columns of weights for x and y stacked horizontally
		stacked_xy_models = np.zeros([output_D,2])

		# Solve the problem for the two image dimensions independently
		X = np.vstack([phase_matrix,prior_matrix])
		for i in [0,1] :
			Y = np.vstack([locations_matrix[:,i],np.zeros([D,1])])
			solution = (np.linalg.lstsq(X,Y))[0]
			stacked_xy_models[:D,i] = solution.flatten()

		return stacked_xy_models
	else:
		return None

# Cut out the relevant section of a results file, between the BEGIN and FINISH markers
def getResultsFileData(filename) :
	with open(filename) as file:
		lines = file.readlines()

	# Find the indices of the begin and finish lines
	beginlineno = next(lno for lno,line in enumerate(lines) if "BEGIN" in line)
	finishlineno = next(lno for lno,line in enumerate(lines) if "FINISH" in line)

	startinfo = lines[beginlineno].strip('BEGIN ').rstrip(os.linesep)
	time_per_frame = float(lines[finishlineno].strip("FINISH").rstrip(os.linesep))

	# Get the video name and the detection radius from the begin line
	vidname = lines[beginlineno].split()[1].rsplit('/')[-1].rsplit('.')[0]
	modelname = lines[beginlineno].split()[2].rsplit('/')[-1].rsplit('.')[0]
	radius = float(lines[beginlineno].split()[3])

	# Work out what variables are in the table
	measure_phase = "phase" in startinfo
	measure_ori = "ori" in startinfo
	find_structures = 'STRUCTURES' in startinfo

	if find_structures:
		structures_list = startinfo.split('STRUCTURES')[1].split()
	else:
		structures_list = None

	# Load the rest of the results into a numpy table
	results_table = np.loadtxt(filename,skiprows=beginlineno+1,comments="FINISH")

	return results_table,vidname,modelname,radius,time_per_frame,measure_phase,measure_ori,structures_list


# Get the results data and interpret into a list of dictionaries of values
def getResultsFileDataAsRows(filename) :
	'''
	getResultsFileDataAsRows(filename)

	Read results data from a results file.

	Inputs:
		filename - the name of the results file to readlines

	Outputs:
		dict_rows - List of dictionaries containing results data. Each dictionary corresponds to a frame.
					Each dictionary has the following entries: 'frame', 'y', 'x', 'post', 'visible', 'class', 'ori', 'phase'
		vidname   - Name of the video that was tested.
		modelname - Name of the model used to make predictions.
		radius    - The radius at which the test was conducted.
		time_per_frame - Time for prediction per frame (averaged over whole frame), in seconds.
	'''
	# Use above method to get the data table
	results_table,vidname,modelname,radius,time_per_frame,measure_phase,measure_ori,structures_list = getResultsFileData(filename)

	# Positions of information on output lines
	frame_linepos = 0
	post_linepos = 1
	y_linepos = 2
	x_linepos = 3
	class_linepos = 4
	visible_linepos = 5

	# Work out what variables are in the table
	if measure_phase :
		phase_linepos = 6
	if measure_ori :
		if measure_phase :
			ori_linepos = 7
		else :
			ori_linepos = 6

	if structures_list is not None:
		struct_begin_linepos = 8 # cannot have structures without ori and phase

	# Create list of (independent) empty dictionaries
	dict_rows = [{} for _ in range(len(results_table))]

	# Loop through rows adding data to the dictionary
	for dict_row,table_row in zip(dict_rows,results_table) :
		dict_row['frame'] = int(table_row[frame_linepos])
		dict_row['y'] = int(table_row[y_linepos])
		dict_row['x'] = int(table_row[x_linepos])
		dict_row['post'] = table_row[post_linepos]
		dict_row['visible'] = int(table_row[visible_linepos]) > 0
		dict_row['class'] = int(table_row[class_linepos])
		if measure_ori :
			dict_row['ori'] = table_row[ori_linepos]
		if measure_phase :
			dict_row['phase'] = table_row[phase_linepos]
		if structures_list is not None:
			dict_row['structs'] = {}
			for s,name in enumerate(structures_list):
				col_num = struct_begin_linepos + s*3
				dict_row['structs'][name] = {'visible': int(table_row[col_num]) > 0 , 'y': int(table_row[col_num+1]), 'x':  int(table_row[col_num+2])}


	return dict_rows,vidname,modelname,radius,time_per_frame





def rocDataFromSummaryFile(summary_file_name,generous=False):

	# Load the relevant columns
	frames_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_frames],dtype=float)
	correct_detections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_correct_detections],dtype=float)
	correct_rejections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_correct_rejections],dtype=float)
	miss_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_misses],dtype=float)
	incorrect_detections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_incorrect_detections],dtype=float)
	incorrect_locations_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_incorrect_locations],dtype=float)
	class_confusions_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_class_confusions],dtype=float)
	obscured_detections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_obscured_detections],dtype=float)
	obscured_rejections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_obscured_rejections],dtype=float)
	obscured_class_confusions_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_obscured_class_confusions],dtype=float)
	obscured_incorrect_locations_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_obscured_incorrect_locations],dtype=float)

	positives_data = correct_detections_data + miss_data + incorrect_locations_data + class_confusions_data
	correct_locations_data = correct_detections_data + class_confusions_data

	# Negatives are those frames marked as either not present or obscured
	negatives_data = correct_rejections_data + incorrect_detections_data + obscured_rejections_data + obscured_detections_data + obscured_class_confusions_data + obscured_incorrect_locations_data
	# False postives are any of the above frames where a detection is made somewhere in the image
	false_positives_data = incorrect_detections_data + obscured_detections_data + obscured_class_confusions_data + obscured_incorrect_locations_data

	fpr = ((false_positives_data[negatives_data != 0.0]/negatives_data[negatives_data != 0.0])).mean()
	correct_loc_rate = np.mean(correct_locations_data/positives_data)

	if generous:
		# Negatives are frames marked as not present
		generous_negatives_data = correct_rejections_data + incorrect_detections_data
		# False postives are any of the above frames where a detection was made somewhere
		generous_false_positives_data = incorrect_detections_data

		fpr_generous = ((generous_false_positives_data[generous_negatives_data != 0.0]/generous_negatives_data[generous_negatives_data != 0.0])).mean()

		return correct_loc_rate, fpr, fpr_generous
	else:
		return correct_loc_rate ,fpr

def accuracyFromSummaryFile(summary_file_name,find_std=False):

	# Load the relevant columns
	correct_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_correct_detections],dtype=float)
	incorrect_locations_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_incorrect_locations],dtype=float)
	class_confusions_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_class_confusions],dtype=float)
	misses_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_misses],dtype=float)

	time_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_time],dtype=float)
	ori_error_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_ori_error],dtype=float)
	phase_error_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_phase_error],dtype=float)

	# Calculate averages
	# To find the 'accuracy', work out what fraction of those frames with positive labels
	# were detected with the correct one
	class_detection_accuracy = (correct_data/(correct_data+incorrect_locations_data+misses_data+class_confusions_data))
	accuracy = class_detection_accuracy.mean()
	time = time_data.mean()*1000.0
	ori_error = np.nanmean(ori_error_data)
	phase_error = np.nanmean(phase_error_data)

	if find_std:
		accuracy_std = class_detection_accuracy.std()
		time_std = time_data.std()*1000.0
		ori_error_std = np.nanstd(ori_error_data)
		phase_error_std = np.nanstd(phase_error_data)
		return accuracy, time, ori_error, phase_error, accuracy_std, time_std, ori_error_std, phase_error_std
	else:
		return accuracy, time, ori_error, phase_error


# Accuracy considering the background class to be just like any other class
# NB have kind of ignored incorrect locations here - they are just considered
# incorrect
def fourClassAccuracyFromSummaryFile(summary_file_name):

	# Load the relevant columns
	frames_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_frames],dtype=float)
	correct_detections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_correct_detections],dtype=float)
	correct_rejections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_correct_rejections],dtype=float)
	incorrect_locations_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_incorrect_locations],dtype=float)
	class_confusions_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_class_confusions],dtype=float)
	misses_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_misses],dtype=float)
	incorrect_detections_data = np.loadtxt(summary_file_name,skiprows=1,usecols=[sum_col_incorrect_detections],dtype=float)

	# Total non-obscured frames
	non_obscured_data = correct_detections_data + misses_data + incorrect_locations_data + class_confusions_data + correct_rejections_data + incorrect_detections_data

	# How many of these frames had the four class label correct
	label_correct_data = correct_detections_data + correct_rejections_data

	return (label_correct_data/non_obscured_data).mean()


def gatherAccuracyStats(experiment_file_name,resultsdir,summary_file_name="summary") :
	model_list = []
	time_list = []
	accuracy_list = []
	ori_error_list = []
	phase_error_list = []

	# Loop through the lines in the experiment definition file
	with open(experiment_file_name) as experiment_file :
		for experimentline in experiment_file :

			modelname = experimentline.split()[0]
			model_list.append(modelname)

			# Determine the name of the corresponding summary file
			summaryname = os.path.join( resultsdir , modelname , summary_file_name)

			accuracy, time, ori_error, phase_error = accuracyFromSummaryFile(summaryname)

			accuracy_list.append(accuracy)
			time_list.append(time)
			ori_error_list.append(ori_error)
			phase_error_list.append(phase_error)

	return model_list, time_list, accuracy_list, ori_error_list, phase_error_list

# Gather a table of accuracy statistics for a list of training experiments and a
# list of testing experiments. The results are returned in a list of lists, with
# testing experiments down the first dimension and training experiments down the
# second. This table is transposed if 'transpose' is True
def gatherAccuracyStatsTrainTest(train_experiment_file_name,test_experiment_file_name,resultsdir,summary_file_name="summary",transpose=False,find_std=False,filtername=None) :
	train_model_list = []
	test_model_list = []
	time_table = []
	accuracy_table = []
	ori_error_table = []
	phase_error_table = []
	if find_std:
		time_std_table = []
		accuracy_std_table = []
		ori_error_std_table = []
		phase_error_std_table = []

	# Read in the list of training experiments
	with open(train_experiment_file_name,'r') as train_experiment_file :
		train_model_list = [line.split()[0] for line in train_experiment_file]

	# Loop through the lines in the test experiment definition file
	with open(test_experiment_file_name,'r') as test_experiment_file:
		for test_experiment_line in test_experiment_file:

			test_model_name = test_experiment_line.split()[0]
			test_model_list.append(test_model_name)

			time_row = []
			accuracy_row = []
			ori_error_row = []
			phase_error_row = []
			if find_std:
				time_std_row = []
				accuracy_std_row = []
				ori_error_std_row = []
				phase_error_std_row = []

			# Loop through lines in the train experiment definition file
			for train_model_name in train_model_list :

				# Determine the name of the corresponding summary file
				if filtername is None:
					summaryname = os.path.join( resultsdir , test_model_name , train_model_name , summary_file_name)
				else:
					summaryname = os.path.join( resultsdir , test_model_name , train_model_name , filtername , summary_file_name)


				if find_std:
					accuracy, time, ori_error, phase_error, accuracy_std, time_std, ori_error_std, phase_error_std = accuracyFromSummaryFile(summaryname,True)
					accuracy_std_row.append(accuracy_std)
					time_std_row.append(time_std)
					ori_error_std_row.append(ori_error_std)
					phase_error_std_row.append(phase_error_std)
				else:
					accuracy, time, ori_error, phase_error = accuracyFromSummaryFile(summaryname)

				accuracy_row.append(accuracy)
				time_row.append(time)
				ori_error_row.append(ori_error)
				phase_error_row.append(phase_error)

			accuracy_table.append(accuracy_row)
			time_table.append(time_row)
			ori_error_table.append(ori_error_row)
			phase_error_table.append(phase_error_row)
			if find_std:
				accuracy_std_table.append(accuracy_std)
				time_std_table.append(time_std)
				ori_error_std_table.append(ori_error_std)
				phase_error_std_table.append(phase_error_std)

	if transpose:
		time_table = list(map(list,zip(*time_table)))
		accuracy_table = list(map(list,zip(*accuracy_table)))
		ori_error_table = list(map(list,zip(*ori_error_table)))
		phase_error_table = list(map(list,zip(*phase_error_table)))
		if find_std:
			time_std_table = list(map(list,zip(*time_std_table)))
			accuracy_std_table = list(map(list,zip(*accuracy_std_table)))
			ori_error_std_table = list(map(list,zip(*ori_error_std_table)))
			phase_error_std_table = list(map(list,zip(*phase_error_std_table)))

	if find_std:
		return train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table, time_std_table, accuracy_std_table, ori_error_std_table, phase_error_std_table
	else:
		return train_model_list, test_model_list, time_table, accuracy_table, ori_error_table, phase_error_table


def getStructureDataFromSummaryFile(filename):
	with open(filename,'r') as sum_file:
		lines = sum_file.readlines()[1:] # skip the header line

	# Split the lines into columns and remvoe data not related to structures
	splitlines = [l.rstrip('\n').split()[sum_col_struct_first:] for l in lines]

	num_cols = len(splitlines[0])
	if num_cols == 0:
		print "ERROR: No structures found in the summary file:", filename
	if not all([len(l) == num_cols for l in splitlines]):
		print "ERROR: lines in summary file",filename,"are not the same length"
		return
	if num_cols % sum_num_cols_per_struct != 0:
		print "ERROR: Could not understand columns in summary file",filename
		return

	structure_names = []
	distance_error = {}
	true_pos_rate = {}
	false_pos_rate = {}

	for s in range(num_cols//sum_num_cols_per_struct):
		struct_start = s*sum_num_cols_per_struct
		names = [line[struct_start] for line in splitlines]

		# Check all the names are the same
		if names.count(names[0]) != len(names):
			print "ERROR: Structure names do not match in summary file",filename
			return

		# Now we know the structure name and can add it to the list
		name = names[0]
		structure_names.append(name)

		# Read in the raw data for this structure
		distance_error_array = np.array([ (np.nan if line[struct_start+sum_struct_distance_col_offset] == 'nan' else float(line[struct_start+sum_struct_distance_col_offset])) for line in splitlines])
		correct_detections_array = np.array([ (float(line[struct_start+sum_struct_correct_detections_col_offset])) for line in splitlines])
		correct_rejections_array = np.array([ (float(line[struct_start+sum_struct_correct_rejections_col_offset])) for line in splitlines])
		incorrect_detections_array = np.array([ (float(line[struct_start+sum_struct_incorrect_detections_col_offset])) for line in splitlines])
		misses_array = np.array([ (float(line[struct_start+sum_struct_misses_col_offset])) for line in splitlines])

		distance_error[name] = np.nanmean(distance_error_array)
		true_pos_rate[name] = np.nanmean(correct_detections_array/(correct_detections_array+misses_array))
		false_pos_rate[name] = np.nanmean(incorrect_detections_array/(incorrect_detections_array+correct_rejections_array))

	return distance_error,true_pos_rate,false_pos_rate

def getStructureDataTrainTest(train_experiment_file_name,test_experiment_file_name,filtername,resultsdir,summary_file_name="summary",transpose=False):#,find_std=False)
	distance_error_table = []
	true_pos_table = []
	false_pos_table = []
	train_model_list = []
	test_model_list = []

	# Read in the list of training experiments
	with open(train_experiment_file_name,'r') as train_experiment_file :
		train_model_list = [line.split()[0] for line in train_experiment_file]

	# Loop through the lines in the test experiment definition file
	with open(test_experiment_file_name,'r') as test_experiment_file:
		for test_experiment_line in test_experiment_file:

			test_model_name = test_experiment_line.split()[0]
			test_model_list.append(test_model_name)

			dist_row = []
			tpr_row = []
			fpr_row = []

			# Loop through lines in the train experiment definition file
			for train_model_name in train_model_list :

				# Determine the name of the corresponding summary file
				summaryname = os.path.join( resultsdir , test_model_name , train_model_name , filtername , summary_file_name)

				# Read info from this summary file
				dist, tpr, fpr = getStructureDataFromSummaryFile(summaryname)

				dist_row.append(dist)
				tpr_row.append(tpr)
				fpr_row.append(fpr)

			distance_error_table.append(dist_row)
			true_pos_table.append(tpr_row)
			false_pos_table.append(fpr_row)

	if transpose:
		distance_error_table = list(map(list,zip(*distance_error_table)))
		true_pos_table = list(map(list,zip(*true_pos_table)))
		false_pos_table = list(map(list,zip(*false_pos_table)))

	return distance_error_table, true_pos_table, false_pos_table



# Read a dataset file
def readDataset(filename) :
	# Use above method to get the data table
	data_table = np.loadtxt(filename,skiprows=3,dtype=str)

	# Create list of (independent) empty dictionaries
	dict_rows = [{} for _ in range(len(data_table))]

	# Loop through rows adding data to the dictionary
	for dict_row,table_row in zip(dict_rows,data_table) :
		dict_row['vid'] = table_row[ds_vidCol]
		dict_row['frame'] = int(table_row[ds_frameCol])
		dict_row['radius'] = int(table_row[ds_radiusCol])
		dict_row['x'] = int(table_row[ds_centrexCol])
		dict_row['y'] = int(table_row[ds_centreyCol])
		dict_row['view'] = int(table_row[ds_viewCol])
		dict_row['ori'] = int(table_row[ds_oriCol])
		dict_row['phase'] = float(table_row[ds_phaseCol])

	return dict_rows
