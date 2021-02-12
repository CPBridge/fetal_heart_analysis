#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <RIFeatures/RIFeatExtractor.hpp>
#include <canopy/classifier/discreteDistribution.hpp>
#include <canopy/classifier/classifier.hpp>
#include <canopy/circularRegressor/circularRegressor.hpp>
#include "thesisUtilities.h"
#include "histograms.h"
#include "rectangleFilterDefines.h"
#include "imageFeatureProcessor.h"
#include "displayFunctions.h"
#include "squareTestingFunctors.h"
#include "particleFilterPosClassJoinedOri.hpp"
#include "particleFilterPosClassJoinedOriPhase.hpp"
#include "particleFilterJoinedStructsPCA.hpp"
#include "particleFilterJoinedSingleStructs.hpp"
#include "mp_tuple_repeat.hpp"

// Constants

int main( int argc, char** argv )
{
	// Namespaces
	using namespace cv;
	using namespace std;
	namespace po = boost::program_options;
	namespace fs = boost::filesystem;
	namespace ut = thesisUtilities;
	namespace cp = canopy;

	// Constants
	constexpr float C_DETECTION_THRESHOLD = 0.0;
	constexpr int C_DISPLAY_MODE_NO_IMPOSE = 4;
	constexpr int C_DISPLAY_MODE_GROUNDTRUTH = 3;
	constexpr int C_DISPLAY_MODE_ALL = 2;
	constexpr int C_DISPLAY_MODE_DETECTION = 1;
	constexpr int C_DISPLAY_MODE_NONE = 0;
	const cv::Scalar CLR_HIDDEN(255,255,255); // white
	const cv::Scalar CLR_SUBSTRUCTURES(0,0,255);

	// Fixed parameters
	constexpr int C_N_VIEW_CLASSES = 3; // excluding background
	constexpr int C_N_STRUCTURES = 14;
	constexpr int C_DEFAULT_N_PARTICLES = 1000;

	int n_rotations, display_mode, n_trees, n_tree_levels, n_trees_phase, n_tree_levels_phase, n_trees_subs, n_tree_levels_subs, n_particles;
	vector<int> n_bins;
	float radius, detection_threshold, frame_rate, structure_dist_temp;
	cv::VideoCapture vid_obj;
	vector<cp::classifier<rectangleFilterDefines::NUM_RECT_PARAMS>> forest;
	int winhalfsize, featurehalfsize, subs_featurehalfsize;
	cv::Size mean_shift_size(30,30);
	cv::TermCriteria mean_shift_term_crit(1,100,1.0);
	fs::path videofile, modelname, maskstring, output_file_name, ground_truth_track_filename;
	string feat_str;
	int problem_type_in;
	vector<fs::path> filter_def_files;

	// Feature parameters
	int num_feat_types;
	vector<float> wl;
	vector<ut::featType_t> feat_type;

	//--------------------------------------------------------------------------
	// OPTIONS PARSING
	//--------------------------------------------------------------------------

	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("pause,u", "pause the video until a key press after each frame")
		("problem_type,p", po::value<int>(&problem_type_in)->default_value(0), "0 class, 1 phase, 2 ori, 3 full, 4 substructure, 5 subs+phase")
		("output,o", po::value<fs::path>(&output_file_name), "output detections to this results file")
		("threshold,t", po::value<float>(&detection_threshold)->default_value(C_DETECTION_THRESHOLD), "detection threshold (range 0 1)")
		("videofile,v", po::value<fs::path>(&videofile), "video to process")
		("modelfile,m", po::value<fs::path>(&modelname)->default_value("modelout"), "root name for model definition files")
		("radius,r", po::value<float>(&radius), "radius at which to search for the objects")
		("n_trees,n", po::value<int>(&n_trees)->default_value(-1), "number of trees in the forest to use (if -ve all trained trees used)")
		("n_tree_levels,l", po::value<int>(&n_tree_levels)->default_value(-1), "number of levels in the forest to use (if -ve all trained levels used)")
		("n_trees_phase,N", po::value<int>(&n_trees_phase)->default_value(-1), "number of trees in the phase forest to use (if -ve all trained trees used)")
		("n_tree_levels_phase,L", po::value<int>(&n_tree_levels_phase)->default_value(-1), "number of levels in the phase forest to use (if -ve all trained levels used)")
		("n_trees_subs,S", po::value<int>(&n_trees_subs)->default_value(-1), "number of trees in the substructure forest to use (if -ve all trained trees used)")
		("n_tree_levels_subs,T", po::value<int>(&n_tree_levels_subs)->default_value(-1), "number of levels in the substructure forest to use (if -ve all trained levels used)")
		("n_particles,Q", po::value<int>(&n_particles)->default_value(C_DEFAULT_N_PARTICLES), "number of particles for the particle filtering")
		("structure_distribution_temperature,X", po::value<float>(&structure_dist_temp)->default_value(-1.0), "smooth the node distributions of the structure forests using this temperature (higher -> smoother)")
		("displaymode,d", po::value<int>(&display_mode)->default_value(C_DISPLAY_MODE_DETECTION), "0 no display, 1 detection only, 2 posterior superimposed, 3 ground truth, 4 posterior only")
		("display_hidden,H", "display hidden detections with a white circle")
		("use_ground_truth_position,P", "give orientation and phase estimates for the ground truth location rather than predicted location")
		("groundtruthtrackfile,g",po::value<fs::path>(&ground_truth_track_filename)->default_value("none"), "track file to read for ground truth" )
		("mask,k", po::value<fs::path>(&maskstring)->default_value("none"), "image file containing mask of the ultrasound fan area")
		("filter,f", "filter detections temporally using a particle filer")
		("filterfile,z", po::value<vector<fs::path>>(&filter_def_files)->multitoken(),"particle filter definition files (list all in order)");

	// Parse input variables
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		cout << "Test a video using a random forest and rectangle filters model" << endl;
		cout << desc << endl;
		return EXIT_SUCCESS;
	}

	const bool pause = bool(vm.count("pause"));
	const bool output_detections = bool(vm.count("output"));
	const ut::problemType_t test_problem = ut::problemType_t(problem_type_in);
	const bool test_phase = (test_problem == ut::ptClassPhase) || (test_problem == ut::ptClassOriPhase) || (test_problem == ut::ptSubstructures) || (test_problem == ut::ptSubstructuresPCA);
	const bool test_ori = (test_problem != ut::ptClass) && (test_problem != ut::ptClassPhase);
	const bool use_gro_truth_location = bool(vm.count("use_ground_truth_position"));
	const bool use_particle_filter = bool(vm.count("filter"));
	const bool display_hidden = bool(vm.count("display_hidden"));
	const bool tracking_substructures = (test_problem == ut::ptSubstructures || test_problem == ut::ptSubstructuresPCA);

	if ( (vm.count("radius") != 1) || (vm.count("videofile") != 1) || (vm.count("modelfile") != 1) )
	{
		cerr << "Error: a (single) video file, radius and model file must be provided" << endl;
		return EXIT_FAILURE;
	}

	if( (test_problem != ut::ptClass) && (test_problem != ut::ptClassOri) && (test_problem != ut::ptClassPhase) && (test_problem != ut::ptClassOriPhase) && (test_problem != ut::ptSubstructures) && (test_problem != ut::ptSubstructuresPCA))
	{
		cerr << "ERROR: The specified problem type is not yet implemented" << endl;
		return EXIT_FAILURE;
	}
	if( (test_problem == ut::ptClass) || (test_problem == ut::ptClassPhase) )
	{
		cerr << "ERROR: The only available problem types are class-orientation (" << ut::ptClassOri << ") and class-orientation-phase (" << ut::ptClassOriPhase << ")." << endl;
		return EXIT_FAILURE;
	}
	if( tracking_substructures && !use_particle_filter )
	{
		cerr << "ERROR: Structures detection is only available with the particle filtering option." << endl;
		return EXIT_FAILURE;
	}


	// Check that we are not trying to use the "ground truth location" flag with
	// the particle filter
	if(use_gro_truth_location && use_particle_filter)
	{
		cerr << "ERROR: Cannot use the ground truth location with particle filtering." << endl;
		return EXIT_FAILURE;
	}

	if( (detection_threshold) > 1.0 || (detection_threshold < 0.0) )
	{
		cerr << "ERROR: Invalid detection threshold: " << detection_threshold << endl;
		return EXIT_FAILURE;
	}

	//--------------------------------------------------------------------------
	// LOAD FOREST MODELS
	//--------------------------------------------------------------------------

	// Add extension to tree file name
	if(modelname.extension().empty())
		modelname.concat(".tr");

	// Attempt to load the tree structure from file
	forest.resize(1);
	const fs::path first_forest_file_name = modelname.parent_path() / modelname.stem().concat("_rotation_0").replace_extension(modelname.extension());
	forest[0].readFromFile(first_forest_file_name.string(),n_trees,n_tree_levels);
	if(!forest[0].isValid())
	{
		cerr << "ERROR: Invalid tree file: " << first_forest_file_name << endl;
		return EXIT_FAILURE;
	}
	// Set the callback function on the forest
	const int n_classes = forest[0].getNumberClasses();
	std::vector<string> class_names;
	forest[0].getClassNames(class_names);
	forest[0].getFeatureDefinitionString(feat_str);

	// Atempt to parse the first line of the definition
	int first_ori_ind;
	if(!parseSquareFeatureDefinitionString(feat_str, first_ori_ind, n_rotations, num_feat_types, winhalfsize, featurehalfsize, feat_type, wl, n_bins))
	{
		cerr << "Error reading features from model file " <<  first_forest_file_name << endl;
		return EXIT_FAILURE;
	}

	if(first_ori_ind != 0)
	{
		cerr << "Error reading features from model file " <<  first_forest_file_name << ", rotation index mismatch in the first file" << endl;
		return EXIT_FAILURE;
	}

	// Load in the phase prediction for the first rotation
	vector<vector<cp::circularRegressor<rectangleFilterDefines::NUM_RECT_PARAMS>>> phase_forest;
	if(test_phase)
	{
		phase_forest.resize(n_rotations);
		for(int r = 0; r < n_rotations; ++r)
			phase_forest[r].resize(n_classes-1);
		for(int c = 1; c < n_classes; ++c)
		{
			const fs::path phase_model_name = modelname.parent_path() / modelname.stem().concat("_rotation_0_phase_" + std::to_string(c)).replace_extension(modelname.extension());
			phase_forest[0][c-1].readFromFile(phase_model_name.string(),n_trees_phase,n_tree_levels_phase);
			if(!phase_forest[0][c-1].isValid())
			{
				cerr << " ERROR: invalid tree input: " << phase_model_name << endl;
				return EXIT_FAILURE;
			}
			// Check that the features match
			string phase_feature_string;
			phase_forest[0][c-1].getFeatureDefinitionString(phase_feature_string);
			if(phase_feature_string != feat_str)
			{
				cerr << " ERROR: features used by " << phase_model_name << " do not match those in the base model" << endl;
				return EXIT_FAILURE;
			}
		}
	}

	// Load in the substructures forest for the first rotation
	vector<cp::classifier<rectangleFilterDefines::NUM_RECT_PARAMS>> subs_forest;
	vector<string> subs_class_names;
	if(tracking_substructures)
	{
		subs_forest.resize(n_rotations);
		const fs::path subs_model_name = modelname.parent_path() / modelname.stem().concat("_subs_rotation_0").replace_extension(modelname.extension());
		subs_forest[0].readFromFile(subs_model_name.string(), n_trees_subs,n_tree_levels_subs);
		subs_forest[0].getClassNames(subs_class_names);
		if(!subs_forest[0].isValid())
		{
			cerr << "ERROR: invalid forest file: " << subs_model_name << endl;
			return EXIT_FAILURE;
		}

		// Check the features match
		string subs_feature_string;
		subs_forest[0].getFeatureDefinitionString(subs_feature_string);
		vector<int> n_bins_temp;
		vector<float> wl_temp;
		vector<ut::featType_t> feat_type_temp;
		int num_feat_types_temp, ori_ind_temp, n_rotations_temp, winhalfsize_temp;
		if(!parseSquareFeatureDefinitionString(subs_feature_string, ori_ind_temp, n_rotations_temp, num_feat_types_temp, winhalfsize_temp, subs_featurehalfsize, feat_type_temp, wl_temp, n_bins_temp))
		{
			cerr << "Error reading features from model file " << subs_model_name << endl;
			return EXIT_FAILURE;
		}
		if(
			(ori_ind_temp != 0)
			|| (num_feat_types_temp != num_feat_types)
			|| (n_rotations_temp != n_rotations)
			|| (feat_type_temp != feat_type)
			|| (winhalfsize_temp != winhalfsize)
			|| (wl_temp != wl)
			|| (n_bins_temp != n_bins)
		)
		{
			cerr << "Feature definition in file " << subs_model_name << " does not match that in the other forests" << endl;
			return EXIT_FAILURE;
		}
		if(structure_dist_temp > 0.0)
			subs_forest[0].raiseNodeTemperature(structure_dist_temp);
	}
	const int n_subs_classes = tracking_substructures ? subs_forest[0].getNumberClasses() : 0 ;

	// Loop through the remaining forests
	forest.resize(n_rotations);
	for(int r = 1; r < n_rotations; ++r)
	{
		// Temporary variables for this forest only
		vector<int> n_bins_temp;
		vector<float> wl_temp;
		vector<ut::featType_t> feat_type_temp;
		int num_feat_types_temp, ori_ind_temp, n_rotations_temp, winhalfsize_temp, featurehalfsize_temp;
		string this_rotation_feature_string;
		const fs::path this_forest_file_name = modelname.parent_path() / modelname.stem().concat("_rotation_" + std::to_string(r)).replace_extension(modelname.extension());

		// Read in this forest
		forest[r].readFromFile(this_forest_file_name.string(),n_trees,n_tree_levels);
		forest[r].getFeatureDefinitionString(this_rotation_feature_string);

		if(!forest[r].isValid())
		{
			cerr << "ERROR: Invalid tree file: " << this_forest_file_name << endl;
			return EXIT_FAILURE;
		}

		// Interpret this string
		if(!parseSquareFeatureDefinitionString(this_rotation_feature_string, ori_ind_temp, n_rotations_temp, num_feat_types_temp, winhalfsize_temp, featurehalfsize_temp, feat_type_temp, wl_temp, n_bins_temp))
		{
			cerr << "Error reading features from model file" << this_forest_file_name <<  endl;
			return EXIT_FAILURE;
		}

		// Ensure all parameters match the first forest
		if( (num_feat_types_temp != num_feat_types)
			|| (ori_ind_temp != r)
			|| (n_rotations_temp != n_rotations)
			|| (winhalfsize_temp != winhalfsize)
			|| (featurehalfsize_temp != featurehalfsize)
			|| (feat_type_temp != feat_type)
			|| (wl_temp != wl)
			|| (n_bins_temp != n_bins)
		  )
		{
			cerr << "Feature definition in file " << this_forest_file_name << " does not match that in the other forests" << endl;
			return EXIT_FAILURE;
		}

		// Read in phase forests for this rotation
		if(test_phase)
		{
			for(int c = 1; c < n_classes; ++c)
			{
				const fs::path phase_model_name = modelname.parent_path() / modelname.stem().concat("_rotation_" + std::to_string(r) + "_phase_" + to_string(c) ).replace_extension(modelname.extension());
				phase_forest[r][c-1].readFromFile(phase_model_name.string(),n_trees_phase,n_tree_levels_phase);
				if(!phase_forest[r][c-1].isValid())
				{
					cerr << " ERROR: invalid tree input!" << phase_model_name << endl;
					return EXIT_FAILURE;
				}
				// Check that the features match
				string phase_feature_string;
				phase_forest[r][c-1].getFeatureDefinitionString(phase_feature_string);
				if(phase_feature_string != this_rotation_feature_string)
				{
					cerr << " ERROR: features used by " << phase_model_name << " do not match those in the base model" << endl;
					return EXIT_FAILURE;
				}
			}
		}

		if(tracking_substructures)
		{
			const fs::path this_rotation_subs_model_name = modelname.parent_path() / modelname.stem().concat("_subs_rotation_" + std::to_string(r) ).replace_extension(modelname.extension());
			subs_forest[r].readFromFile(this_rotation_subs_model_name.string(), n_trees_subs,n_tree_levels_subs);
			if(!subs_forest[r].isValid())
			{
				cerr << "ERROR: invalid forest file: " << this_rotation_subs_model_name << endl;
				return EXIT_FAILURE;
			}

			// Check features match
			string this_rotation_subs_feature_string;
			subs_forest[r].getFeatureDefinitionString(this_rotation_subs_feature_string);
			if(!parseSquareFeatureDefinitionString(this_rotation_subs_feature_string, ori_ind_temp, n_rotations_temp, num_feat_types_temp, winhalfsize_temp, featurehalfsize_temp, feat_type_temp, wl_temp, n_bins_temp))
			{
				cerr << "ERROR: Invalid feature definition string in model " << this_rotation_subs_model_name << endl;
				return EXIT_FAILURE;
			}
			if( (num_feat_types_temp != num_feat_types)
				|| (ori_ind_temp != r)
				|| (n_rotations_temp != n_rotations)
				|| (winhalfsize_temp != winhalfsize)
				|| (featurehalfsize_temp != subs_featurehalfsize)
				|| (feat_type_temp != feat_type)
				|| (wl_temp != wl)
				|| (n_bins_temp != n_bins)
			  )
			{
				cerr << "Feature definition in file " << this_rotation_subs_model_name << " does not match that in the other forests" << endl;
				return EXIT_FAILURE;
			}

			// Also check subs classes match
			vector<string> subs_class_names_temp;
			subs_forest[r].getClassNames(subs_class_names_temp);
			if(subs_class_names_temp != subs_class_names)
			{
				cerr << "Names of structure classes in file " << this_rotation_subs_model_name << " does not match those in the other forests" << endl;
				return EXIT_FAILURE;
			}

			if(structure_dist_temp > 0.0)
				subs_forest[0].raiseNodeTemperature(structure_dist_temp);
		}
	}

	const double angle_step = 2.0*M_PI/n_rotations;

	vector<int> n_bins_vector;
	for(int ft = 0 ; ft < num_feat_types; ++ft)
	{
		if(feat_type[ft] != ut::ftInt)
			n_bins_vector.push_back(n_bins[ft]);
	}
	const int num_vector_feat_types = n_bins_vector.size();
	const bool using_scalar_feat = std::any_of(feat_type.cbegin(),feat_type.cend(),[](const ut::featType_t f){return f == ut::ftInt;});

	//--------------------------------------------------------------------------
	// OPEN THE VIDEO FILE
	//--------------------------------------------------------------------------

	// Open the video
	vid_obj.open(videofile.string());
	if ( !vid_obj.isOpened() )
	{
		cerr  << "Could not open reference " << videofile << endl;
		return EXIT_FAILURE;
	}
	frame_rate = vid_obj.get(cv::CAP_PROP_FPS);

	// Get the frame rate
	if(isnan(frame_rate))
	{
		frame_rate = ut::getFrameRate(videofile.string(),videofile.parent_path().string());
		if(isnan(frame_rate))
		{
			cerr << "Could not determine frame rate forn the video " << endl;
			return EXIT_FAILURE;
		}
	}

	const int xsize = vid_obj.get(cv::CAP_PROP_FRAME_WIDTH);
	const int ysize = vid_obj.get(cv::CAP_PROP_FRAME_HEIGHT);
	int n_frames = vid_obj.get(cv::CAP_PROP_FRAME_COUNT);
	const float scale_factor = float(winhalfsize)/radius;
	radius *= scale_factor;
	const int radius_int = std::ceil(radius);
	const int xresize = int(float(xsize)*scale_factor);
	const int yresize = int(float(ysize)*scale_factor);

	//--------------------------------------------------------------------------
	// CREATE RECTANGULAR FILTER FEATURE EXTRACTION OBJECTS
	//--------------------------------------------------------------------------

	const unsigned motion_feat_index = distance(feat_type.cbegin(),find(feat_type.cbegin(), feat_type.cend(),ut::ftMotion));

	// Create objects for extracting feature representations
	vector<imageFeatureProcessor> feat_processor(num_feat_types);
	for(int ft = 0; ft < num_feat_types; ++ft)
		feat_processor[ft].initialise(feat_type[ft],yresize,xresize,wl[ft],frame_rate);

	// Create the functor object
	squareTestingFunctorMixed test_ftr(using_scalar_feat,num_vector_feat_types,n_bins_vector);

	// Wrapper with the main featurehalfsize
	auto test_ftr_main_wrapper = [&] (const auto id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params)
	{
		return test_ftr(id,params,featurehalfsize);
	};

	// Wrapper with the structures featurehalfsize
	auto test_ftr_subs_wrapper = [&] (const auto id, const std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params)
	{
		return test_ftr(id,params,subs_featurehalfsize);
	};

	//--------------------------------------------------------------------------
	// READ IN MASK AND GET LISTS OF VALID PIXELS
	//--------------------------------------------------------------------------

	const double feature_to_win_ratio = (featurehalfsize == winhalfsize) ? 1.0 : double(featurehalfsize + 0.5) / double(winhalfsize + 0.5);
	const double subs_feature_to_win_ratio = (subs_featurehalfsize == winhalfsize) ? 1.0 : double(subs_featurehalfsize + 0.5) / double(winhalfsize + 0.5);

	// Attempt to load a mask
	Mat_<unsigned char> valid_mask;
	if( !ut::prepareMask(maskstring.string(),Size(xsize,ysize),valid_mask,std::ceil(feature_to_win_ratio*radius/scale_factor),Size(xresize,yresize), featurehalfsize) )
	{
		cerr << "ERROR loading the mask: " << maskstring << endl;
		return EXIT_FAILURE;
	}

	// Attempt to load a mask for substructure detection
	Mat_<unsigned char> subs_valid_mask;
	if(tracking_substructures)
	{
		if( !ut::prepareMask(maskstring.string(),Size(xsize,ysize),subs_valid_mask,std::ceil(subs_feature_to_win_ratio*radius/scale_factor),Size(xresize,yresize), subs_featurehalfsize) )
		{
			cerr << "ERROR loading the mask: " << maskstring << endl;
			return EXIT_FAILURE;
		}
	}

	// Get a list of valid pixels to test
	vector<cv::Point> valid_pixels;
	vector<int> img_to_valid;
	ut::findValidPixels(valid_pixels,valid_mask,1,radius_int,&img_to_valid);

	//--------------------------------------------------------------------------
	// Set Up Particle Filters
	//--------------------------------------------------------------------------

	vector<cp::discreteDistribution> output_posts;
	vector<vector<Mat_<float>>> post(n_rotations);
	particleFilterPosClassJoinedOri<C_N_VIEW_CLASSES> p_filt_ori;
	particleFilterPosClassJoinedOriPhase<C_N_VIEW_CLASSES> p_filt_ori_phase;
	particleFilterJoinedStructsPCA<C_N_VIEW_CLASSES> p_filt_subs_pca;
	particleFilterJoinedSingleStructs<C_N_VIEW_CLASSES,C_N_STRUCTURES> p_filt_subs;
	if(use_particle_filter)
	{
		switch(test_problem)
		{
			case ut::ptAbdomen:
			case ut::ptClass :
				// Not implemented
				break;
			case ut::ptClassOri :
				if(filter_def_files.size() != 1)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori = particleFilterPosClassJoinedOri<C_N_VIEW_CLASSES>(yresize, xresize, n_particles, radius, filter_def_files[0].string(), &valid_mask);
				if (!p_filt_ori.checkInit())
				{
					cerr << "Error reading filter definition files " << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori.initialiseParticles();
				break;
			case ut::ptClassPhase :
				// Not implemented
				break;
			case ut::ptClassOriPhase :
				if(filter_def_files.size() != 2)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori_phase = particleFilterPosClassJoinedOriPhase<C_N_VIEW_CLASSES>(yresize, xresize, n_particles, radius, frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), &valid_mask);
				if (!p_filt_ori_phase.checkInit())
				{
					cerr << "Error reading filter definition file " << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori_phase.initialiseParticles();
				break;
			case ut::ptSubstructures :
				if(filter_def_files.size() != 3)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_subs = particleFilterJoinedSingleStructs<C_N_VIEW_CLASSES,C_N_STRUCTURES>(yresize,xresize,n_particles,radius,frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), filter_def_files[2].string(), subs_class_names, &valid_mask,&subs_valid_mask);
				if (!p_filt_subs.checkInit())
				{
					cerr << "Error reading filter definition file " << endl;
					return EXIT_FAILURE;
				}
				p_filt_subs.initialiseParticles();
				break;
			case ut::ptSubstructuresPCA :
				if(filter_def_files.size() != 3)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_subs_pca = particleFilterJoinedStructsPCA<C_N_VIEW_CLASSES>(yresize,xresize,n_particles,radius,frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), filter_def_files[2].string(), subs_class_names, &valid_mask,&subs_valid_mask);
				if (!p_filt_subs_pca.checkInit())
				{
					cerr << "Error reading filter definition file " << endl;
					return EXIT_FAILURE;
				}
				p_filt_subs_pca.initialiseParticles();
				break;
		}
	}

	//--------------------------------------------------------------------------
	// CREATE IMAGES TO HOLD DETECTION POSTERIORS
	//--------------------------------------------------------------------------

	else // Not using particle filter
	{
		// Allocate memory for the detection posteriors
		output_posts.resize(valid_pixels.size());
		for(int p = 0; p < int(valid_pixels.size()); ++p)
			output_posts[p].initialise(n_classes);

		// Set up images to to contain the output probabilities, and windows to display them
		post.resize(n_rotations);
		for(int r = 0; r < n_rotations; ++r)
		{
			post[r] = vector<Mat_<float>>(n_classes-1);
			for(int c = 1; c < n_classes; ++c)
			{
				post[r][c-1] = Mat_<float>::zeros(yresize,xresize);
			}
		}
	}

	const bool using_motion = any_of(feat_type.begin(), feat_type.end(), [](ut::featType_t ft) { return (ft == ut::ftMotion); }) ;

	//--------------------------------------------------------------------------
	// VARIABLES NEEDED TO HOLD SUBSTRUCTURE LOCATIONS
	//--------------------------------------------------------------------------

	// Arrays to hold the locations of the substructures
	vector<Point> subs_locations;
	std::array<double,C_N_STRUCTURES> subs_x_arr, subs_y_arr;
	std::array<structVisible_enum,C_N_STRUCTURES> subs_visible_arr;
	std::fill(subs_visible_arr.begin(),subs_visible_arr.end(),structVisible_enum::svVisible);
	if(tracking_substructures)
	{
		subs_locations.resize(n_subs_classes-1);
	}

	//--------------------------------------------------------------------------
	// SET UP DISPLAY WINDOWS AND VARIABLES
	//--------------------------------------------------------------------------

	// Display window for the input stream
	if(display_mode != C_DISPLAY_MODE_NONE)
	{
		namedWindow( "Input Stream", WINDOW_AUTOSIZE );
		moveWindow("Input Stream", 0,0);
	}
	if(display_mode == C_DISPLAY_MODE_GROUNDTRUTH)
	{
		namedWindow( "Ground Truth", WINDOW_AUTOSIZE );
		moveWindow("Ground Truth", xsize,0);
	}
	vector<vector<Mat_<Vec3b>>> post_superimposed;
	if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
	{
		// With particle filters, there is one display image per class
		if(use_particle_filter)
		{
			post_superimposed.resize(1);
			if(tracking_substructures)
			{
				post_superimposed[0].resize(n_subs_classes-1);
				for(int c = 1; c < n_subs_classes; ++c)
				{
					namedWindow("Detection Posterior " + subs_class_names[c], WINDOW_AUTOSIZE );
					moveWindow("Detection Posterior " + subs_class_names[c], (c/2)*xsize,(c%2)*ysize);
				}
			}
			else
			{
				post_superimposed[0].resize(n_classes-1);
				for(int c = 1; c < n_classes; ++c)
				{
					namedWindow("Detection Posterior " + class_names[c], WINDOW_AUTOSIZE );
					moveWindow("Detection Posterior " + class_names[c], (c/2)*xsize,(c%2)*ysize);
				}
			}
		}

		// Without particle filters there are multiple display images per class,
		// One for each rotation
		else
		{
			post_superimposed.resize(n_rotations);
			for(int r = 0; r < n_rotations; ++r)
			{
				post_superimposed[r].resize(n_classes-1);
				for(int c = 1; c < n_classes; ++c)
				{
					namedWindow("Detection Posterior " + class_names[c] + " rotation " + std::to_string(r*angle_step), WINDOW_AUTOSIZE );
					moveWindow("Detection Posterior " + class_names[c] + " rotation " + std::to_string(r*angle_step), (c/2)*xsize,(c%2)*ysize);
				}
			}
		}
	}

	//--------------------------------------------------------------------------
	// READ IN THE GROUND TRUTH IF REQUIRED
	//--------------------------------------------------------------------------

	// Read in the ground truth trackfile
	bool gro_tru_headup;
	int gro_tru_radius;
	std::vector<bool> gro_tru_labelled;
	std::vector<int> gro_tru_x, gro_tru_y, gro_tru_ori_degrees, gro_tru_view, gro_tru_phase_point;
	std::vector<float> gro_tru_phase;
	std::vector<ut::heartPresent_t> gro_tru_present;
	if( display_mode == C_DISPLAY_MODE_GROUNDTRUTH || use_gro_truth_location)
	{
		if(ground_truth_track_filename.string().compare("none") == 0)
		{
			cerr << "ERROR: Ground truth display or ground truth location results requested but no track file provided (use -g option)" << endl;
			return EXIT_FAILURE;
		}

		if(!ut::readTrackFile(ground_truth_track_filename.string(), n_frames, gro_tru_headup, gro_tru_radius, gro_tru_labelled, gro_tru_present,
						   gro_tru_y, gro_tru_x, gro_tru_ori_degrees, gro_tru_view, gro_tru_phase_point, gro_tru_phase) )
		{
			cerr << "Error reading the requested track file " << ground_truth_track_filename << endl;
			return EXIT_FAILURE;
		}
	}

	//--------------------------------------------------------------------------
	// SET UP FILE OUTPUT
	//--------------------------------------------------------------------------

	// Open the output file and output a "begin" line
	ofstream output_file;
	if(output_detections)
	{
		output_file.open(output_file_name.string().c_str());
		if (!output_file.is_open())
		{
			cerr << "ERROR: Could not open file " << output_file_name << " for writing" << endl;
			return EXIT_FAILURE;
		}

		output_file << "BEGIN" << " " << videofile << " " << modelname.string() << " " << radius/scale_factor << " ";
		switch(test_problem)
		{
			case ut::ptClass:
				output_file << "weight y x class\n";
				break;
			case ut::ptClassOri:
				output_file << "weight y x class ori\n";
				break;
			case ut::ptClassPhase:
				output_file << "weight y x class phase\n";
				break;
			case ut::ptClassOriPhase:
				output_file << "weight y x class phase ori\n";
				break;
			case ut::ptSubstructures:
			case ut::ptSubstructuresPCA:
				output_file << "weight y x class phase ori STRUCTURES";
				for(auto name_it = subs_class_names.cbegin()+1 ; name_it != subs_class_names.cend() ; ++name_it)
					output_file << " " << *name_it;
				output_file << '\n';
				break;
			case ut::ptAbdomen:
				/* Not supported */
				output_file << '\n';
				break;
		}
	}

	//--------------------------------------------------------------------------
	// SET UP CALLBACKS FOR THE PARTICLE FILTERS
	//--------------------------------------------------------------------------

	auto pos_class_ori_reweight_lambda = [&] (auto first_point, const auto last_point, auto class_it, auto ori_it, auto weight_it)
	mutable
	{
		// Loop over points individually
		while(first_point != last_point)
		{
			// Find the two closest orientation bins to use
			const double quotient = *ori_it/angle_step;
			const int lower_ori_bin = std::floor(quotient);
			const int upper_ori_bin = (lower_ori_bin == n_rotations-1) ? 0 : lower_ori_bin + 1;

			// Use the forest models to score the hypotheses at these two orientations
			float lower_score, upper_score;
			forest[lower_ori_bin].probabilitySingle(first_point, first_point+1, class_it, &lower_score, true, test_ftr_main_wrapper);
			forest[upper_ori_bin].probabilitySingle(first_point, first_point+1, class_it, &upper_score, true, test_ftr_main_wrapper);

			// Value between 0 and 1 indicating fractional position between upper and lower bins
			const double lower_distance = quotient - lower_ori_bin;

			// Use weighted average of the two two scores as final score to update weight
			(*weight_it) *= (lower_score*(1.0-lower_distance) + upper_score*lower_distance);

			// Increment iterators
			++first_point;
			++class_it;
			++ori_it;
			++weight_it;
		}
	};

	auto phase_reweight_lambda = [&] (const int c, auto first_pos_state, const auto last_pos_state, auto phase_it, auto weight_it)
	mutable
	{
		// Loop over points individually
		while(first_pos_state != last_pos_state)
		{
			const statePosClassOri& state = *first_pos_state;

			// Find the two closest orientation bins to use
			const double quotient = state.ori/angle_step;
			const int lower_ori_bin = std::floor(quotient);
			const int upper_ori_bin = (lower_ori_bin == n_rotations-1) ? 0 : lower_ori_bin + 1;

			// Use the forest models to score the hypotheses at these two orientations
			float lower_score, upper_score;
			phase_forest[lower_ori_bin][c-1].probabilitySingle(first_pos_state, first_pos_state+1, phase_it, &lower_score, true, test_ftr_main_wrapper);
			phase_forest[upper_ori_bin][c-1].probabilitySingle(first_pos_state, first_pos_state+1, phase_it, &upper_score, true, test_ftr_main_wrapper);

			// Value between 0 and 1 indicating fractional position between upper and lower bins
			const double lower_distance = quotient - lower_ori_bin;

			// Use weighted average of the two two scores as final score to update weight
			(*weight_it) *= (lower_score*(1.0-lower_distance) + upper_score*lower_distance);

			// Increment iterators
			++first_pos_state;
			++phase_it;
			++weight_it;
		}
	};

	auto subs_reweight_lambda = [&] (auto first_point, const auto last_point, auto class_it, auto heart_pos_it, auto weight_it, const bool single_label)
	mutable
	{
		// Loop over points individually
		while(first_point != last_point)
		{
			// Find the two closest orientation bins to use
			const double quotient = (*heart_pos_it).ori/angle_step;
			const int lower_ori_bin = std::floor(quotient);
			const int upper_ori_bin = (lower_ori_bin == n_rotations-1) ? 0 : lower_ori_bin + 1;

			// Use the forest models to score the hypotheses at these two orientations
			float lower_score, upper_score;
			subs_forest[lower_ori_bin].probabilitySingle(first_point, first_point+1, class_it, &lower_score, true, test_ftr_subs_wrapper);
			subs_forest[upper_ori_bin].probabilitySingle(first_point, first_point+1, class_it, &upper_score, true, test_ftr_subs_wrapper);

			// Value between 0 and 1 indicating fractional position between upper and lower bins
			const double lower_distance = quotient - lower_ori_bin;

			// Use weighted average of the two two scores as final score to update weight
			(*weight_it) = (lower_score*(1.0-lower_distance) + upper_score*lower_distance);

			// Increment iterators
			++first_point;
			++heart_pos_it;
			++weight_it;
			if(!single_label)
				++class_it;
		}
	};

	auto pos_class_ori_lambda_tuple = std::make_tuple(pos_class_ori_reweight_lambda);
	auto pos_class_phase_ori_lambda_tuple = std::make_tuple(pos_class_ori_reweight_lambda,phase_reweight_lambda);
	auto pca_subs_lambda_tuple = std::make_tuple(pos_class_ori_reweight_lambda,phase_reweight_lambda,subs_reweight_lambda);
	auto single_subs_lambda_tuple = std::tuple_cat(std::make_tuple(pos_class_ori_reweight_lambda,phase_reweight_lambda,subs_reweight_lambda),ut::mp_tuple_repeat<C_N_STRUCTURES>(subs_reweight_lambda));

	//--------------------------------------------------------------------------
	// START THE LOOP
	//--------------------------------------------------------------------------

	// Loop through frames
	double frame_start = clock();
	const double test_start = (double) getTickCount();
	for(int f = 0 ; f < n_frames; ++f)
	{
		Mat_<Vec3b> disp, gro_tru_disp;
		Mat_<unsigned char> I, I_resize;
		bool heart_visible = true;
		int max_class, overall_max_point_x, overall_max_point_y;
		double overall_max_val = -1.0;
		float max_phase = 0.0, heart_rate = 0.0;
		double ori;

		//----------------------------------------------------------------------
		// TIMINGS
		//----------------------------------------------------------------------

		// Timers to time since the start of the previous loop
		// in order to monitor frame rate
		const double frame_stop = (double) getTickCount();
		const double achieved_fps = getTickFrequency()/double(frame_stop - frame_start);
		frame_start = (double) getTickCount();

		//----------------------------------------------------------------------
		// READ FRAME
		//----------------------------------------------------------------------

		// Get frame
		vid_obj >> disp;

		// Need to check this because sometimes OpenCV's estimate
		// of the number of frames is wrong
		if(disp.empty())
		{
			n_frames = f;
			break;
		}
		cvtColor(disp,I,cv::COLOR_BGR2GRAY);

		if(scale_factor != 1.0)
			resize(I,I_resize,Size(xresize,yresize));
		else
			I_resize = I; // shallow copy

		//----------------------------------------------------------------------
		// TRANSFORM IMAGE
		//----------------------------------------------------------------------

		// Transform the image to the desired representation
		// and place into the feature extractor
		for(int ft = 0; ft < num_feat_types; ++ft)
		{
			if(feat_type[ft] == ut::ftInt)
				cv::integral(I_resize,test_ftr.getScalarImageRef());
			else
			{
				Mat_<float> Imag, Iori;
				feat_processor[ft].extract(I_resize,Imag,Iori);
				test_ftr.setVectorImages(ft,Imag,Iori);
			}
		}

		// If we are using motion features, we can't do anything more on the first frame
		if( (f == 0) && using_motion)
			continue;

		//----------------------------------------------------------------------
		// UPDATE PARTICLE FILTERS
		//----------------------------------------------------------------------

		if(use_particle_filter)
		{
			// Prepare the display images for the visualisation functions
			if(display_mode == C_DISPLAY_MODE_ALL )
			{
				for(auto& post : post_superimposed[0])
					disp.copyTo(post);
			}
			else if (display_mode == C_DISPLAY_MODE_NO_IMPOSE)
			{
				for(auto& post : post_superimposed[0])
					post = Mat_<Vec3b>::zeros(ysize,xsize);
			}

			switch(test_problem)
			{
				case ut::ptAbdomen:
				case ut::ptClass :
				// not implemented
				break;

				case ut::ptClassOri :
				{
					if(using_motion)
						p_filt_ori.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pos_class_ori_lambda_tuple);
					else
						p_filt_ori.step(I_resize,pos_class_ori_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
					 	p_filt_ori.visualiseOri(post_superimposed[0].data());
					std::tuple<statePosClassOri> class_pos_ori_state;
					p_filt_ori.meanShiftEstimate(class_pos_ori_state,overall_max_val);
					max_class = std::get<0>(class_pos_ori_state).c;
					ori = std::get<0>(class_pos_ori_state).ori;
					heart_visible = std::get<0>(class_pos_ori_state).visible;
					overall_max_point_x = int(std::get<0>(class_pos_ori_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(class_pos_ori_state).y/scale_factor);
				}
				break;

				case ut::ptClassPhase :
				// not implemented
				break;

				case ut::ptClassOriPhase :
				{

					if(using_motion)
						p_filt_ori_phase.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pos_class_phase_ori_lambda_tuple);
					else
						p_filt_ori_phase.step(I_resize,pos_class_phase_ori_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
					 	p_filt_ori_phase.visualiseOriPhase(post_superimposed[0].data());
					std::tuple<statePosClassOri,statePhase> class_pos_phase_ori_state;
					p_filt_ori_phase.meanShiftEstimate(class_pos_phase_ori_state,overall_max_val);
					max_class = std::get<0>(class_pos_phase_ori_state).c;
					max_phase = std::get<1>(class_pos_phase_ori_state).ph;
					heart_rate = std::get<1>(class_pos_phase_ori_state).ph_rate*(60.0/(2.0*M_PI));
					ori = std::get<0>(class_pos_phase_ori_state).ori;
					heart_visible = std::get<0>(class_pos_phase_ori_state).visible;
					overall_max_point_x = int(std::get<0>(class_pos_phase_ori_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(class_pos_phase_ori_state).y/scale_factor);
				}
				break;

			case ut::ptSubstructures :
				{
					if(using_motion)
						p_filt_subs.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),single_subs_lambda_tuple);
					else
						p_filt_subs.step(I_resize,single_subs_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
						p_filt_subs.visualiseSubstructures(post_superimposed[0].data());
					typename particleFilterJoinedSingleStructs<C_N_VIEW_CLASSES,C_N_STRUCTURES>::combined_state_type subs_state;
					p_filt_subs.meanShiftEstimate(subs_state,overall_max_val);
					max_class = std::get<0>(subs_state).c;
					max_phase = std::get<1>(subs_state).ph;
					heart_rate = std::get<1>(subs_state).ph_rate*(60.0/(2.0*M_PI));
					ori = std::get<0>(subs_state).ori;
					heart_visible = std::get<0>(subs_state).visible;
					overall_max_point_x = int(std::get<0>(subs_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(subs_state).y/scale_factor);
					p_filt_subs.structPositionArray(subs_state,subs_y_arr,subs_x_arr,subs_visible_arr);
					for(int s = 0; s < n_subs_classes-1; ++s)
					{
						subs_locations[s] = Point(subs_x_arr[s]/scale_factor,subs_y_arr[s]/scale_factor);
					}

				}
				break;

				case ut::ptSubstructuresPCA :
				{
					if(using_motion)
						p_filt_subs_pca.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pca_subs_lambda_tuple);
					else
						p_filt_subs_pca.step(I_resize,pca_subs_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
						p_filt_subs_pca.visualiseSubstructures(post_superimposed[0].data());
					std::tuple<statePosClassOri,statePhase,stateSubstructuresPCA<C_N_VIEW_CLASSES>> subs_state;
					p_filt_subs_pca.meanShiftEstimate(subs_state,overall_max_val);
					max_class = std::get<0>(subs_state).c;
					max_phase = std::get<1>(subs_state).ph;
					heart_rate = std::get<1>(subs_state).ph_rate*(60.0/(2.0*M_PI));
					ori = std::get<0>(subs_state).ori;
					heart_visible = std::get<0>(subs_state).visible;
					overall_max_point_x = int(std::get<0>(subs_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(subs_state).y/scale_factor);
					for(int s = 0; s < n_subs_classes-1; ++s)
					{
						subs_locations[s] = Point(std::get<2>(subs_state).x[s]/scale_factor,std::get<2>(subs_state).y[s]/scale_factor);
						subs_visible_arr[s] = std::get<2>(subs_state).visible[s];
					}

				}
				break;
			} // end switch
		} // end if particle filter

		//----------------------------------------------------------------------
		// UNFILTERED CASE
		//----------------------------------------------------------------------

		else // not using particle filter
		{
			int gro_tru_valid_id = 0;
			cv::Point gro_tru_point_resize(0,0);
			const int gro_tru_ori_index = use_gro_truth_location ? std::lround((float(gro_tru_ori_degrees[f])*M_PI/180.0)/angle_step) % n_rotations : 0;
			if(use_gro_truth_location)
			{
				gro_tru_point_resize.x = gro_tru_x[f]*scale_factor;
				gro_tru_point_resize.y = gro_tru_y[f]*scale_factor;
				gro_tru_valid_id = img_to_valid[gro_tru_point_resize.y*xresize + gro_tru_point_resize.x];

				// If this results in a location that is no valid, use the img_to_valid
				// array to find the nearest valid location
				if(gro_tru_valid_id < 0)
				{
					gro_tru_valid_id = -(gro_tru_valid_id+1);
					gro_tru_point_resize = valid_pixels[gro_tru_valid_id];
				}
			}

			float max_val_at_gro_tru_location = -1.0;
			int max_ori_index, max_r_at_gro_tru_location;
			cv::Point detection_point_resize;

			// Loop over the forests in different orientations
			for(int r = 0; r < n_rotations; ++r)
			{
				// Use random forests for detection
				forest[r].predictDistSingle(valid_pixels.cbegin(), valid_pixels.cend(), output_posts.begin(), test_ftr_main_wrapper);

				#pragma omp parallel for
				for(int c = 1; c < n_classes; ++c)
				{
					double min_val, max_val;
					cv::Point min_point, max_point;
					cv::Rect window_rect;

					for(unsigned p = 0; p < valid_pixels.size(); ++p)
						post[r][c-1](valid_pixels[p]) = output_posts[p].pdf(c);

					// Use mean shift to find the detection location
					minMaxLoc(post[r][c-1],&min_val,&max_val,&min_point,&max_point);
					window_rect = Rect(max_point,mean_shift_size);
					meanShift(post[r][c-1],window_rect,mean_shift_term_crit);

					#pragma omp critical
					{
						if(max_val > overall_max_val)
						{
							overall_max_val = max_val;
							detection_point_resize.x = (window_rect.x+window_rect.width/2.0);
							detection_point_resize.y = (window_rect.y+window_rect.height/2.0);
							max_class = c;
							max_ori_index = r;
						}
					}

					// Superimpose the detection posterior over the input image
					if( (display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE) )
					{
						Mat colour_channels[3];
						colour_channels[0] = Mat::zeros(post[r][c-1].rows,post[r][c-1].cols,CV_8U);
						colour_channels[1] = Mat::zeros(post[r][c-1].rows,post[r][c-1].cols,CV_8U);
						colour_channels[2] = 255*post[r][c-1];
						colour_channels[2].convertTo(colour_channels[2],CV_8U);
						merge(colour_channels,3,post_superimposed[r][c-1]);
						resize(post_superimposed[r][c-1],post_superimposed[r][c-1],Size(xsize,ysize));
						if(display_mode == C_DISPLAY_MODE_ALL)
							post_superimposed[r][c-1] = post_superimposed[r][c-1] + disp;
					}
				}

				if(use_gro_truth_location)
				{
					if(output_posts[gro_tru_valid_id].pdf(gro_tru_view[f]) > max_val_at_gro_tru_location)
					{
						max_val_at_gro_tru_location = output_posts[gro_tru_valid_id].pdf(gro_tru_view[f]);
						max_r_at_gro_tru_location = r;
					}
				}
			}
			overall_max_point_x = detection_point_resize.x/scale_factor;
			overall_max_point_y = detection_point_resize.y/scale_factor;

			// Check that this is a valid position
			if(detection_point_resize.x <= radius_int) detection_point_resize.x = radius_int + 1;
			if(detection_point_resize.x >= xresize-radius_int) detection_point_resize.x = xresize - radius_int - 1;
			if(detection_point_resize.y <= radius_int) detection_point_resize.y = radius_int + 1;
			if(detection_point_resize.y >= yresize-radius_int) detection_point_resize.y = yresize - radius_int - 1;

			// Average over the neighbouring orientation bins
			double sin_ori = 0.0, cos_ori = 0.0;
			if(test_ori)
			{
				const cv::Point point_to_use = (use_gro_truth_location ? gro_tru_point_resize : detection_point_resize);
				const int class_to_use = (use_gro_truth_location ? gro_tru_view[f] : max_class) - 1;

				for(int offset = -1; offset <= 1; ++offset)
				{
					int ori_bin = (use_gro_truth_location ? max_r_at_gro_tru_location : max_ori_index) + offset;

					// Wrap around
					if(ori_bin == -1)
					 	ori_bin = n_rotations - 1;
					if(ori_bin == n_rotations)
						ori_bin = 0;

					const float weight = post[ori_bin][class_to_use](point_to_use);
					sin_ori += weight*std::sin(ori_bin*angle_step);
					cos_ori += weight*std::cos(ori_bin*angle_step);
				}
			}
			ori = test_ori ? std::atan2(sin_ori,cos_ori) : 0.0;

			// Make any other predictions that were requested
			switch(test_problem)
			{
				case ut::ptAbdomen:
				case ut::ptClass :
				case ut::ptClassOri :
					// Nothing more to do
					break;
				case ut::ptClassPhase :
				case ut::ptClassOriPhase :
				{
					cp::vonMisesDistribution phase_dist;
					if(use_gro_truth_location)
						phase_forest[gro_tru_ori_index][gro_tru_view[f]-1].predictDistSingle(&gro_tru_point_resize,(&gro_tru_point_resize)+1,&phase_dist, test_ftr_main_wrapper);
					else
						phase_forest[max_ori_index][max_class-1].predictDistSingle(&detection_point_resize,(&detection_point_resize)+1,&phase_dist, test_ftr_main_wrapper);
					max_phase = phase_dist.getMu();
				}
				break;
				case ut::ptSubstructures :
				case ut::ptSubstructuresPCA :

					break;
			}
		} // end not using particle filter

		//----------------------------------------------------------------------
		// OUTPUT RESULTS
		//----------------------------------------------------------------------

		// Output the detections if desired
		if(output_detections)
		{
			output_file << f << " " ;
			output_file << overall_max_val << " " << overall_max_point_y << " " << overall_max_point_x << " " << max_class << " " << heart_visible;

			if(test_problem == ut::ptClassPhase || test_problem == ut::ptClassOriPhase || test_problem == ut::ptSubstructuresPCA || test_problem == ut::ptSubstructures)
				output_file << " " << max_phase;

			if(test_problem == ut::ptClassOri || test_problem == ut::ptClassOriPhase || test_problem == ut::ptSubstructuresPCA || test_problem == ut::ptSubstructures)
				output_file << " " << ori;

			if(test_problem == ut::ptSubstructuresPCA || test_problem == ut::ptSubstructures)
			{
				for(int s = 0; s < n_subs_classes-1; ++s )
					output_file << " " << (subs_visible_arr[s] == structVisible_enum::svVisible ? 1 : 0) << " " << subs_locations[s].y << " " << subs_locations[s].x;
			}

			output_file << '\n';
		}

		//----------------------------------------------------------------------
		// DISPLAY
		//----------------------------------------------------------------------

		// Display the ground truth if desired
		if(display_mode == C_DISPLAY_MODE_GROUNDTRUTH)
		{
			gro_tru_disp = disp.clone();

			// Draw circle for position of the detection
			if(gro_tru_labelled[f] && (gro_tru_present[f] != ut::hpNone) )
				displayHeart(gro_tru_disp, gro_tru_x[f],gro_tru_y[f], gro_tru_view[f], float(gro_tru_ori_degrees[f])*M_PI/180.0, gro_tru_phase[f], gro_tru_present[f], test_problem, radius/scale_factor);

			// Display the main image
			imshow("Ground Truth", gro_tru_disp);
		}

		// Display the imput image with detection annotations
		if(display_mode != C_DISPLAY_MODE_NONE)
		{
			if(!pause)
				putText(disp,string("FPS ") + to_string(achieved_fps),Point(5,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,127,127) );

			if( (overall_max_val >= detection_threshold) && (heart_visible || display_hidden) )
			{
				displayHeart(disp, overall_max_point_x, overall_max_point_y, max_class, ori, max_phase, heart_visible ? ut::hpPresent : ut::hpNone, test_problem, radius/scale_factor);

				if(use_particle_filter && (test_problem == ut::ptClassOriPhase  || test_problem == ut::ptSubstructures || test_problem == ut::ptSubstructuresPCA))
					putText(disp,string("BPM ") + to_string(int(heart_rate)),Point(overall_max_point_x + (0.5*radius/scale_factor)*std::cos(ori), overall_max_point_y - (1.3*radius/scale_factor)*std::sin(-ori) ),FONT_HERSHEY_PLAIN,1.0,Scalar(0,127,127) );
			}
			// Display the substructure detections
			if(tracking_substructures)
			{
				const cv::Scalar subs_colour = heart_visible ? CLR_SUBSTRUCTURES : CLR_HIDDEN;
				for(int s = 1; s < n_subs_classes; ++s)
				{
					if(subs_visible_arr[s-1] == structVisible_enum::svVisible)
						circle(disp,subs_locations[s-1],3,subs_colour,1);
				}
			}
			imshow("Input Stream", disp);
		}
		if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
		{
			if(use_particle_filter)
			{
				if(tracking_substructures)
				{
					for(int c = 1; c < n_subs_classes; ++c)
						imshow("Detection Posterior " + subs_class_names[c],post_superimposed[0][c-1]);
				}
				else
				{
					for(int c = 1; c < n_classes; ++c)
						imshow("Detection Posterior " + class_names[c],post_superimposed[0][c-1]);
				}
			}
			else
			{
				for(int r = 0; r < n_rotations; ++r)
					for(int c = 1; c < n_classes; ++c)
						imshow("Detection Posterior " + class_names[c] + " rotation " + std::to_string(r*angle_step),post_superimposed[r][c-1]);
			}
		}

		if(pause)
			waitKey(0);
		else
			waitKey(1);

	}

	const double test_stop = (double) getTickCount();

	if(output_detections)
	{
		output_file << "FINISH " << double(test_stop - test_start)/(getTickFrequency()*n_frames) << '\n';
		output_file.close();
	}

}
