#!/usr/bin/python

import sys			 # argv
import os.path
import argparse as ap  # parser (for arguments)
import substrdatasetfromtracks as subsmodule
import heartdatasets as heartmodule
from CPBUtils import getTracksList


# Parse command line arguements
parser = ap.ArgumentParser(description='Create a dataset from a set of trackfiles')
parser.add_argument('subs_track_directory',help="directory containing the substructure track files")
parser.add_argument('heart_track_directory',help="directory containing the heart track files")
parser.add_argument('radius_frac',help="fraction of the heart radius at which to detect structures, either specify a literal value or input Jmax and J values for rotation invariant features separated by a slash, e.g. 3/5")
parser.add_argument('substructures_file',help="name of the file containing substructures to use")
parser.add_argument('--mask_directory','-m',help="directory containing the image masks",default='')
parser.add_argument('--cross_val','-c',action='store_true',help="Generate datasets for a cross validation",)
parser.add_argument('--n_examples_per_struct','-n',type=int,help="number of examples per substructure for the subs model",default=100)
parser.add_argument('--n_examples_per_view','-p',type=int,help="number of examples per view for the heart model",default=100)
parser.add_argument('--n_views','-v',type=int,help="number of view types labelled")
parser.add_argument('--outfilename','-o',help="name of the output file",default="dataset")
parser.add_argument('--exclude_list','-e',help="patient names to exclude from the dataset",default=[],nargs='*')
parser.add_argument('--first_frame','-f',type=int,help="avoid frames before this frame",default=0)
parser.add_argument('--overlap_frac','-l',type=float,help="fraction of the radius that must be within the masked area for a background sample",default=1.0)
parser.add_argument('--separate_classes','-s',help="create a different dataset for the subsructures in each class",action="store_true")
args = parser.parse_args()

# Find substructures modelfilename by appending "_subs"
path = os.path.dirname(args.outfilename)
basename,ext = os.path.splitext(os.path.basename(args.outfilename))

# Work out the radius fraction
if '/' in args.radius_frac :
	jmax = int((args.radius_frac).split('/')[0])
	nj = int((args.radius_frac).split('/')[1])
	radius_frac = float(jmax+1)/float(nj)
else :
	# Treat this as a literal float
	radius_frac = float(args.radius_frac)

if args.cross_val:
	# Get a list of all the trackfiles
	_,trackfiles = getTracksList(args.heart_track_directory,'.tk')

	# Find the unique patients in this list
	patients = [os.path.basename(f.rsplit('_',1)[0]) for f in trackfiles]
	unique_patients = list(set(patients))

	# Create one dataset excluding each of the patients
	for excluded_patient in unique_patients:
		this_dataset_name = os.path.join( path , basename + '_ex' + excluded_patient)
		if args.separate_classes :
			for v in range(1,args.n_views+1) :
				subsoutname = this_dataset_name + "_subs" + str(v) + ext
				subsmodule.makeSubsDataset(args.subs_track_directory,args.heart_track_directory,radius_frac,args.substructures_file,v,args.mask_directory,args.n_examples_per_struct,subsoutname,[excluded_patient],args.first_frame,args.overlap_frac,list_heart_radius=True)
		else :
			subsoutname = this_dataset_name + "_subs" + ext
			subsmodule.makeSubsDataset(args.subs_track_directory,args.heart_track_directory,radius_frac,args.substructures_file,-1,args.mask_directory,args.n_examples_per_struct,subsoutname,[excluded_patient],args.first_frame,args.overlap_frac,list_heart_radius=True)

		# Create the heart model
		heartoutname = this_dataset_name + ext
		heartmodule.makeHeartDataset(args.heart_track_directory,args.n_views,args.mask_directory,args.n_examples_per_view,heartoutname,[excluded_patient],args.first_frame,args.overlap_frac)

else:
	# Create the subs dataset(s)
	# Either do one for all substructures, or break down by class
	if args.separate_classes :
		for v in range(1,args.n_views+1) :
			subsoutname = os.path.join( path , basename + "_subs" + str(v) + ext)
			subsmodule.makeSubsDataset(args.subs_track_directory,args.heart_track_directory,radius_frac,args.substructures_file,v,args.mask_directory,args.n_examples_per_struct,subsoutname,args.exclude_list,args.first_frame,args.overlap_frac,list_heart_radius=True)
	else :
		subsoutname = os.path.join( path , basename + "_subs" + ext)
		subsmodule.makeSubsDataset(args.subs_track_directory,args.heart_track_directory,radius_frac,args.substructures_file,-1,args.mask_directory,args.n_examples_per_struct,subsoutname,args.exclude_list,args.first_frame,args.overlap_frac,list_heart_radius=True)

	# Create the heart model
	heartmodule.makeHeartDataset(args.heart_track_directory,args.n_views,args.mask_directory,args.n_examples_per_view,args.outfilename,args.exclude_list,args.first_frame,args.overlap_frac)
