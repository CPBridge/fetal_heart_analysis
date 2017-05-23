import argparse as ap

# This is intended to replicate the functionality of the argument parser in
# test.cpp

test_arg_parser = ap.ArgumentParser()

test_arg_parser.add_argument('--pause','-u', help="pause the video until a key press after each frame")
test_arg_parser.add_argument('--display_hidden','-H', help="display hidden detections with a white circle")
test_arg_parser.add_argument('--displaymode','-d', type=int, default=1, help="0 no display, 1 detection only, 2 posterior superimposed, 3 ground truth, 4 posterior only")
test_arg_parser.add_argument('--record','-c', default="none", help="record an output video with the given filename")
test_arg_parser.add_argument('--videofile','-v', help="video to process")
test_arg_parser.add_argument('--modelfile','-m', default="modelout", help="root name for model definition files")
test_arg_parser.add_argument('--radius','-r', type=float, help="radius at which to search for the objects")
test_arg_parser.add_argument('--mask','-k', help="image file containing mask of the ultrasound fan area")
test_arg_parser.add_argument('--n_trees','-n', type=int, default=-1, help="number of trees in the forest to use (if -ve all trained trees used)")
test_arg_parser.add_argument('--n_tree_levels','-l', type=int, default=-1, help="number of levels in the forest to use (if -ve all trained levels used)")
test_arg_parser.add_argument('--n_trees_phase','-N', type=int, default=-1, help="number of trees in the phase forest to use (if -ve all trained trees used)")
test_arg_parser.add_argument('--n_tree_levels_phase','-L', type=int, default=-1, help="number of levels in the phase forest to use (if -ve all trained levels used)")
test_arg_parser.add_argument('--n_trees_subs','-S', type=int, default=-1, help="number of trees in the substructure forest to use (if -ve all trained trees used)")
test_arg_parser.add_argument('--n_tree_levels_subs','-T', type=int, default=-1, help="number of levels in the substructure forest to use (if -ve all trained levels used)")
test_arg_parser.add_argument('--n_particles','-Q', type=int, default=1000, help="number of particles for the particle flitering")
test_arg_parser.add_argument('--threshold','-t', type=float,default=0.0, help="detection threshold (range 0 1)")
test_arg_parser.add_argument('--problem_type','-p', type=int, default=0, help="0 class, 1 phase, 2 ori, 3 full, 4 substructure, 5 subs+phase")
test_arg_parser.add_argument('--output','-o', help="output detections to this results file")
test_arg_parser.add_argument('--filter','-f', action='store_true',help="filter detections temporally using a particle filter")
test_arg_parser.add_argument('--use_ground_truth_position','-P', action='store_true', help="give orientation and phase estimates for the ground truth location rather than predicted location")
test_arg_parser.add_argument('--ri_feature_method','-R', help="method to use for calculating rotation invariant features (default auto)")
test_arg_parser.add_argument('--ri_coupling_method','-C', help="method to use for coupling within invariant features (default auto)")
test_arg_parser.add_argument('--spatial_feature_memoisation','-M', help="use spatial memoisation within feature extractors")
test_arg_parser.add_argument('--structure_distribution_temperature','-X', type=float,default=-1.0, help="smooth the node distributions of the structure forests using this temperature (higher -> smoother)")
test_arg_parser.add_argument('--filterfile','-z', nargs='*', help="particle filter definition files (list all in order)")
test_arg_parser.add_argument('--groundtruthtrackfile','-g', help="track file to read for ground truth" )
test_arg_parser.add_argument('--groundtruthsubsfile','-i', help="track file to read for substructures ground truth" )
test_arg_parser.add_argument('--groundtruthabdfile','-j', help="track file to read for abdomen ground truth" )
test_arg_parser.add_argument('--pad_border_fraction','-b', type=float,default=0.0, help="fraction of the image width and height that are added to each side to define the limits of the Hough space")
test_arg_parser.add_argument('--pixel_skip','-q', type=int, default=1, help="stride of pixels at which to evaluate the hough Forest forest")
test_arg_parser.add_argument('--motion_tracking','-x', help="Use motion estimate to track abdominal movement ")
test_arg_parser.add_argument('--scale','-s', type=float, help="scale factor to apply before processing");
