import argparse as ap

# This is intended to replicate the functionality of the argument parser in
# train_precompute.cpp

C_DEFAULT_RADIUS=30
C_DEFAULT_TREES=8
C_DEFAULT_LEVELS=10

train_arg_parser = ap.ArgumentParser()

train_arg_parser.add_argument('--dataset','-d',  help="input dataset file")
train_arg_parser.add_argument('--videodirectory','-v', help="directory containing video files")
train_arg_parser.add_argument('--modelfile','-o', help="root name for output files")
train_arg_parser.add_argument('--featureType','-f',nargs='*', help="type of feature used (multiple arguments may be listed")
train_arg_parser.add_argument('--nj','-j',type=int,nargs='*', help="number of radial bases")
train_arg_parser.add_argument('--nk','-k',type=int,nargs='*', help="max basis rotation order")
train_arg_parser.add_argument('--nm','-m',type=int,nargs='*', help="Fourier histogram expansion order")
train_arg_parser.add_argument('--feature_set','-c',nargs='*', help="feature coupling type (basic/couple_simple/couple_extra)")
train_arg_parser.add_argument('--jmax','-x',type=int,nargs='*', help="maximum value of j that is permitted to be used")
train_arg_parser.add_argument('--radius','-r',type=int, default=C_DEFAULT_RADIUS, help="radius at which to detect structures in the trained model")
train_arg_parser.add_argument('--wavelength','-w',type=float,nargs='*', help="monogenic filter centre-wavelength")
train_arg_parser.add_argument('--trees','-n',type=int, default=C_DEFAULT_TREES, help="number of trees")
train_arg_parser.add_argument('--depth','-l',type=int, default=C_DEFAULT_LEVELS, help="number of levels per tree")
train_arg_parser.add_argument('--without_split_node_dists','-s' , help="do not fit node distributions to split nodes in the forest")
train_arg_parser.add_argument('--orientation','-a' , help="also train an orientation prediction model")
train_arg_parser.add_argument('--phase','-p' , help="train an additional phase prediction model")
train_arg_parser.add_argument('--display', help="display training examples")
