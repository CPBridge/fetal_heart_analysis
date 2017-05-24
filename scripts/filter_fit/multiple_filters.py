#!/usr/bin/python

import argparse as ap  # parser (for arguments)
from os import linesep
from fit_class_ori_filter import fit_class_ori_filter
from fit_pca_structs_filter import fit_pca_structs_filter
from fit_partitioned_structs_filter import fit_partitioned_structs_filter

# Set up argument parser
parser = ap.ArgumentParser(description='Produce filters with multiple values of the visibility parameters')
parser.add_argument('track_directory',help="directory containing the track files")
parser.add_argument('outfilenamestem',help="stem for names of the output files")
parser.add_argument('n_views',type=int,help="number of views types labelled (ex background)")
parser.add_argument('mean_shift_params',help="Name of a mean-shift parameters file to append to the offset parameters")
parser.add_argument('patients_file',help="Name of file containing list of patients for cross validation")
parser.add_argument('--type','-T',help='What kind of filter: class_ori [default], pca_structs, partitioned_structs',default='class_ori')
parser.add_argument('--structs_track_directory','-s',help='Directory containing structure annotation files')
parser.add_argument('--structures_list','-l',help='File defining the structures')
parser.add_argument('--weight_precision','-r',type=float,help="weight precision (inverse variance) to use as a regularising term, for structures models only", default=1.0)
parser.add_argument('--windowlength','-W',type=int,help="length of one window in cycles",default=1)
parser.add_argument('--pcacomponents','-p',type=int,help="Number of PCA components, for PCA structs nodel only",default=5)
parser.add_argument('--hidden_equilibrium_fraction','-H',type=float,help='override the hidden equilibrium fraction parameter with this value',nargs='*')
parser.add_argument('--hidden_time_constant','-t',type=float,help='override the hidden time constant parameter with this value',nargs='*')
parser.add_argument('--hidden_weight','-w',type=float,help='override the hidden weight parameter with this value',nargs='*')

args = parser.parse_args()

# Checks on inputs
if args.hidden_equilibrium_fraction is None or len(args.hidden_equilibrium_fraction) < 1:
	print "ERROR: One or more hidden_equilibrium_fractions must be supplied"
	exit()
if args.hidden_time_constant is None or len(args.hidden_time_constant) < 1:
	print "ERROR: One or more hidden_time_constants must be supplied"
	exit()
if args.hidden_weight is None or len(args.hidden_weight) < 1:
	print "ERROR: One or more hidden_weights must be supplied"
	exit()
if not args.type in ['class_ori','pca_structs','partitioned_structs']:
	print "ERROR: type must be 'class_ori','pca_structs', or 'partitioned_structs'"
	exit()
if args.type in ['pca_structs','partitioned_structs'] and (args.structs_track_directory is None) or (args.structures_list is None):
	print "ERROR: if fitting a structures model, you must provides the substructures annotation directory (-s option) and the structures list (-l option)"
	exit()

# Get a list of the patients for cross validation
with open(args.patients_file,'r') as patients_file_obj:
	patients = [l.rstrip(linesep) for l in patients_file_obj]

# Loop over eq fraction
for ef in args.hidden_equilibrium_fraction:
	# Loop over time Constants
	for tc in args.hidden_time_constant:
		# Loop over hidden weights
		for w in args.hidden_weight:

			# The string to use in the name
			paramstr = '_ef' + str(ef) + '_tc' + str(tc) + '_w' + str(w)

			# Loop for cross-validation
			for ex_patient in patients:
				outfilename = args.outfilenamestem + paramstr + '_ex' + ex_patient

				# Call the function
				if args.type == 'class_ori':
					fit_class_ori_filter(args.track_directory,outfilename,args.n_views,mean_shift_params=args.mean_shift_params,excludelist=[ex_patient],hidden_equilibrium_fraction=ef,hidden_time_constant=tc,hidden_weight=w)
				elif args.type == 'pca_structs':
					fit_pca_structs_filter(args.structs_track_directory,args.track_directory,args.structures_list,outfilename=outfilename,nviews=args.n_views,windowlength=args.windowlength,mean_shift_params=args.mean_shift_params,exclude_list=[ex_patient],hidden_equilibrium_fraction=ef,hidden_time_constant=tc,hidden_weight=w,weight_precision=args.weight_precision,pcacomponents=args.pcacomponents)
				elif args.type == 'partitioned_structs':
					fit_partitioned_structs_filter(args.structs_track_directory,args.track_directory,args.structures_list,outfilename=outfilename,nviews=args.n_views,windowlength=args.windowlength,mean_shift_params=args.mean_shift_params,exclude_list=[ex_patient],hidden_equilibrium_fraction=ef,hidden_time_constant=tc,hidden_weight=w,weight_precision=args.weight_precision)
