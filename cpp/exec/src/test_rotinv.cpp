#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <RIFeatures/RIFeatExtractor.hpp>
#include <canopy/classifier/discreteDistribution.hpp>
#include <canopy/classifier/classifier.hpp>
#include "randomForestFunctorBase.hpp"
#include "jointOrientationRegressor.hpp"
#include "jointOriPhaseRegressor.hpp"
#include "circCircSingleRegressor.hpp"
#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include <canopy/circularRegressor/circularRegressor.hpp>
#include "RIHoughForest.hpp"
#include "particleFilterPosClass.hpp"
#include "particleFilterPosClassOri.hpp"
#include "particleFilterPosClassPhase.hpp"
#include "particleFilterPosClassPhaseOri.hpp"
#include "particleFilterSubStructs.hpp"
#include "particleFilterSingleStructs.hpp"
#include "thesisUtilities.h"
#include "imageFeatureProcessor.h"
#include "orientationTestingFunctor.h"
#include "mp_tuple_repeat.hpp"


int main( int argc, char** argv )
{
	//--------------------------------------------------------------------------
	// DECLARATIONS AND CONSTANTS
	//--------------------------------------------------------------------------

	// Namespaces
	using namespace cv;
	using namespace std;
	namespace po = boost::program_options;
	namespace fs = boost::filesystem;
	namespace ut = thesisUtilities;
	namespace cp = canopy;

	// Constants
	constexpr double C_DETECTION_THRESHOLD = 0.0;

	// Particle filtering parameters
	constexpr int C_DEFAULT_N_PARTICLES = 1000;

	// Number of heart structures
	constexpr int C_N_VIEW_CLASSES = 3; // excluding background
	constexpr int C_N_STRUCTURES = 14;

	// View Colours
	const cv::Scalar CLR_ABDOMEN(255,0,255); // magenta
	const cv::Scalar CLR_HIDDEN(255,255,255); // white
	const cv::Scalar CLR_SUBSTRUCTURES(0,0,255);

	// Mean shift window size
	constexpr int C_MEAN_SHIFT_WINSIZE = 31;
	//constexpr double C_GAUSS_BLUR_SIZE = 15.0;

	constexpr int C_DISPLAY_MODE_NO_IMPOSE = 4;
	constexpr int C_DISPLAY_MODE_GROUNDTRUTH = 3;
	constexpr int C_DISPLAY_MODE_ALL = 2;
	constexpr int C_DISPLAY_MODE_DETECTION = 1;
	constexpr int C_DISPLAY_MODE_NONE = 0;

	constexpr float C_DEFAULT_PAD_FRACTION = 0.0;
	constexpr float C_HEART_ABDOMEN_RATIO = 1.8;

	// Declarations
	int radius_int, max_class, n_trees, n_tree_levels, n_trees_phase, n_tree_levels_phase, n_trees_subs, n_tree_levels_subs, train_radius, display_mode, problem_type_in, pixel_skip, n_particles;
	float radius, scale_factor, detection_threshold, frame_rate,pad_fraction, structure_dist_temp;
	const Size mean_shift_size(C_MEAN_SHIFT_WINSIZE,C_MEAN_SHIFT_WINSIZE);
	const TermCriteria mean_shift_term_crit(1,100,1.0);
	fs::path videofile, modelname, maskstring, ground_truth_track_filename, ground_truth_subs_filename, ground_truth_abd_filename, record_vidname, output_file_name;
	string feat_str, subs_feat_str, abd_hough_feat_str, ri_method_string, coupling_method_string;
	VideoWriter output_video;
	vector<string> class_names, subs_class_names;
	vector<fs::path> filter_def_files;

	// Feature parameters
	vector<int> J,K,M, max_rot_order, feature_set_type, basis_type, Jmax;
	vector<float> wl;
	vector<string> feat_type_str;

	//--------------------------------------------------------------------------
	// OPTIONS PARSING
	//--------------------------------------------------------------------------

	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("pause,u", "pause the video until a key press after each frame")
		("display_hidden,H", "display hidden detections with a white circle")
		("displaymode,d", po::value<int>(&display_mode)->default_value(C_DISPLAY_MODE_DETECTION), "0 no display, 1 detection only, 2 posterior superimposed, 3 ground truth, 4 posterior only")
		("record,c", po::value<fs::path>(&record_vidname)->default_value("none"), "record an output video with the given filename")
		("videofile,v", po::value<fs::path>(&videofile), "video to process")
		("modelfile,m", po::value<fs::path>(&modelname)->default_value("modelout"), "root name for model definition files")
		("radius,r", po::value<float>(&radius), "radius at which to search for the objects")
		("mask,k", po::value<fs::path>(&maskstring)->default_value("none"), "image file containing mask of the ultrasound fan area")
		("n_trees,n", po::value<int>(&n_trees)->default_value(-1), "number of trees in the forest to use (if -ve all trained trees used)")
		("n_tree_levels,l", po::value<int>(&n_tree_levels)->default_value(-1), "number of levels in the forest to use (if -ve all trained levels used)")
		("n_trees_phase,N", po::value<int>(&n_trees_phase)->default_value(-1), "number of trees in the phase forest to use (if -ve all trained trees used)")
		("n_tree_levels_phase,L", po::value<int>(&n_tree_levels_phase)->default_value(-1), "number of levels in the phase forest to use (if -ve all trained levels used)")
		("n_trees_subs,S", po::value<int>(&n_trees_subs)->default_value(-1), "number of trees in the substructure forest to use (if -ve all trained trees used)")
		("n_tree_levels_subs,T", po::value<int>(&n_tree_levels_subs)->default_value(-1), "number of levels in the substructure forest to use (if -ve all trained levels used)")
		("n_particles,Q", po::value<int>(&n_particles)->default_value(C_DEFAULT_N_PARTICLES), "number of particles for the particle filtering")
		("threshold,t", po::value<float>(&detection_threshold)->default_value(C_DETECTION_THRESHOLD), "detection threshold (range 0 1)")
		("problem_type,p", po::value<int>(&problem_type_in)->default_value(0), "0 class, 1 phase, 2 ori, 3 full, 4 substructure, 5 subspca")
		("output,o", po::value<fs::path>(&output_file_name), "output detections to this results file")
		("filter,f", "filter detections temporally using a particle filer")
		("use_ground_truth_position,P", "give orientation and phase estimates for the ground truth location rather than predicted location")
		("ri_feature_method,R", po::value<string>(&ri_method_string), "method to use for calculating rotation invariant features (default auto)")
		("ri_coupling_method,C", po::value<string>(&coupling_method_string), "method to use for coupling within invariant features (default auto)")
		("spatial_feature_memoisation,M", "use spatial memoisation within feature extractors")
		("structure_distribution_temperature,X", po::value<float>(&structure_dist_temp)->default_value(-1.0), "smooth the node distributions of the structure forests using this temperature (higher -> smoother)")
		("filterfile,z", po::value<vector<fs::path>>(&filter_def_files)->multitoken(),"particle filter definition files (list all in order)")
		("groundtruthtrackfile,g",po::value<fs::path>(&ground_truth_track_filename)->default_value("none"), "track file to read for ground truth" )
		("groundtruthsubsfile,i",po::value<fs::path>(&ground_truth_subs_filename)->default_value("none"), "track file to read for substructures ground truth" )
		("groundtruthabdfile,j",po::value<fs::path>(&ground_truth_abd_filename)->default_value("none"), "track file to read for abdomen ground truth" )
		("pad_border_fraction,b", po::value<float>(&pad_fraction)->default_value(C_DEFAULT_PAD_FRACTION), "fraction of the image width and height that are added to each side to define the limits of the Hough space")
		("pixel_skip,q", po::value<int>(&pixel_skip)->default_value(1), "stride of pixels at which to evaluate the hough Forest forest")
		("motion_tracking,x", "Use motion estimate to track abdominal movement ")
		("scale,s", po::value<float>(&scale_factor), "scale factor to apply before processing");

	// Parse input variables
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		cout << "Test a video using a random forest and invariant features model" << endl;
		cout << desc << endl;
		return 1;
	}

	const bool use_particle_filter = bool(vm.count("filter"));
	const ut::problemType_t test_problem = ut::problemType_t(problem_type_in);
	const bool pause = bool(vm.count("pause"));
	const bool record = (record_vidname.string().compare("none") != 0);
	const bool display_hidden = bool(vm.count("display_hidden"));
	const bool using_abdomen_hough = (test_problem == ut::ptAbdomen);
	const bool tracking_heart = (test_problem != ut::ptAbdomen);
	const bool use_gro_truth_location = bool(vm.count("use_ground_truth_position"));
	const bool use_memoiser = bool(vm.count("spatial_feature_memoisation"));
	const bool output_detections = bool(vm.count("output"));

	if ( (vm.count("radius") != 1) || (vm.count("videofile") != 1) || (vm.count("modelfile") != 1) )
	{
		cerr << "ERROR: a (single) video file, radius and model file must be provided" << endl;
		return EXIT_FAILURE;
	}

	if( (detection_threshold) > 1.0 || (detection_threshold < 0.0) )
	{
		cerr << "ERROR: Invalid detection threshold: " << detection_threshold << endl;
		return EXIT_FAILURE;
	}

	// Add extension to tree file name
	if(modelname.extension().empty())
		modelname.concat(".tr");

	if(! ( (test_problem == ut::ptClass) || (test_problem == ut::ptClassOri) || (test_problem == ut::ptClassPhase) || (test_problem == ut::ptClassOriPhase) || (test_problem == ut::ptSubstructures) || (test_problem == ut::ptSubstructuresPCA) || (test_problem == ut::ptAbdomen) ) )
	{
		cerr << "ERROR: Unrecognised problem type" << endl;
		return EXIT_FAILURE;
	}

	if( (test_problem == ut::ptAbdomen) && use_particle_filter)
	{
		cerr << "ERROR: Particle filtering with abdomen detection is not yet implemented!" << endl;
		return EXIT_FAILURE;
	}

	// Check that we are not trying to use the "ground truth location" flag with
	// the particle filter
	if(use_gro_truth_location && use_particle_filter)
	{
		cerr << "ERROR: Cannot use the ground truth location with particle filtering." << endl;
		return EXIT_FAILURE;
	}

	RIFeatures::RIFeatExtractor::calculationMethod_enum ri_feature_method;
	if(vm.count("ri_feature_method") < 1)
		ri_feature_method = RIFeatures::RIFeatExtractor::cmAuto;
	else
	{
		if(!RIFeatures::RIFeatExtractor::stringToCalcMethod(ri_method_string,ri_feature_method))
		{
			cerr << "ERROR: Unrecognised feature calculation method: " << ri_method_string << endl;
			return EXIT_FAILURE;
		}
	}

	RIFeatures::RIFeatExtractor::couplingMethod_enum ri_coupling_method;
	if(vm.count("ri_coupling_method") < 1)
		ri_coupling_method = RIFeatures::RIFeatExtractor::comAuto;
	else
	{
		if(!RIFeatures::RIFeatExtractor::stringToCoupleMethod(coupling_method_string,ri_coupling_method))
		{
			cerr << "ERROR: Unrecognised feature calculation method: " << ri_method_string << endl;
			return EXIT_FAILURE;
		}
	}

	//--------------------------------------------------------------------------
	// LOAD FOREST MODELS
	//--------------------------------------------------------------------------

	// Attempt to load the tree structure from file
	cp::classifier<2> forest;
	cp::jointOrientationRegressor<cp::circCircSingleRegressor<orientationTestingFunctor>,2> ori_forest;
	if( (test_problem == ut::ptClassOri) )
	{
		const fs::path ori_model_name = modelname.parent_path() / modelname.stem().concat("_ori").replace_extension(modelname.extension());
		ori_forest.readFromFile(ori_model_name.string(), n_trees,n_tree_levels);
		if(!ori_forest.isValid())
		{
			cerr << " ERROR: invalid tree input! " << ori_model_name << endl;
			return EXIT_FAILURE;
		}
		ori_forest.getFeatureDefinitionString(feat_str);
		ori_forest.getClassNames(class_names);
	}
	else
	{
		forest.readFromFile(modelname.string(), n_trees,n_tree_levels);
		if(!forest.isValid())
		{
			cerr << " ERROR: invalid tree input! " << modelname << endl;
			return EXIT_FAILURE;
		}
		forest.getFeatureDefinitionString(feat_str);
		forest.getClassNames(class_names);
	}

	// Read in phase prediction forest
	vector<cp::circularRegressor<2>> phase_forest;
	if(test_problem == ut::ptClassPhase)
	{
		// There is one phase prediction model for each class
		phase_forest.resize(forest.getNumberClasses()-1);
		for(int c = 1; c < forest.getNumberClasses(); ++c)
		{
			const fs::path phase_model_name = modelname.parent_path() / modelname.stem().concat( "_phase" + to_string(c) ).replace_extension(modelname.extension());
			phase_forest[c-1].readFromFile(phase_model_name.string(),n_trees_phase,n_tree_levels_phase);
			if(!phase_forest[c-1].isValid())
			{
				cerr << " ERROR: invalid tree input!" << phase_model_name << endl;
				return EXIT_FAILURE;
			}
			// Check that the features match
			string temp_feat_str;
			phase_forest[c-1].getFeatureDefinitionString(temp_feat_str);
			if(temp_feat_str != feat_str)
			{
				cerr << " ERROR: features used by " << phase_model_name << " do not match those in the base model" << endl;
				return EXIT_FAILURE;
			}
		}
	}

	// Read in joint orientation/phase prediction forests
	vector<cp::jointOriPhaseRegressor<cp::circCircSingleRegressor<orientationTestingFunctor>,2>> ori_phase_forest;
	const bool using_joint_ori_phase_regressor = (test_problem == ut::ptClassOriPhase || test_problem == ut::ptSubstructures || test_problem == ut::ptSubstructuresPCA );
	if(using_joint_ori_phase_regressor)
	{
		// There is one ori-phase prediction model for each class
		ori_phase_forest.resize(forest.getNumberClasses()-1);
		for(int c = 1; c < forest.getNumberClasses(); ++c)
		{
			const fs::path ori_phase_model_name = modelname.parent_path() / modelname.stem().concat( "_phaseori" + to_string(c) ).replace_extension(modelname.extension());
			ori_phase_forest[c-1].readFromFile(ori_phase_model_name.string(),n_trees_phase,n_tree_levels_phase);
			if(!ori_phase_forest[c-1].isValid())
			{
				cerr << " ERROR: invalid tree input! " << ori_phase_model_name << endl;
				return EXIT_FAILURE;
			}
			// Check that the features match
			string temp_feat_str;
			ori_phase_forest[c-1].getFeatureDefinitionString(temp_feat_str);
			if(temp_feat_str != feat_str)
			{
				cerr << " ERROR: features used by " << ori_phase_model_name << "do not match those in the base model" << endl;
				return EXIT_FAILURE;
			}
		}
	}

	// Load in substructure forests
	cp::classifier<2> subs_forest;
	const bool tracking_substructures = (test_problem == ut::ptSubstructures || test_problem == ut::ptSubstructuresPCA);
	if(tracking_substructures)
	{
		const fs::path subs_model_name = modelname.parent_path() / modelname.stem().concat( "_subs" ).replace_extension(modelname.extension());
		subs_forest.readFromFile(subs_model_name.string(), n_trees_subs,n_tree_levels_subs);
		if(!subs_forest.isValid())
		{
			cerr << " ERROR: invalid tree input! " << subs_model_name << endl;
			return EXIT_FAILURE;
		}
		subs_forest.getFeatureDefinitionString(subs_feat_str);
		subs_forest.getClassNames(subs_class_names);
		if(structure_dist_temp > 0.0)
			subs_forest.raiseNodeTemperature(structure_dist_temp);
	}

	// Load in abdomen hough forest
	cp::RIHoughForest<orientationTestingFunctor,2> abd_hough_forest;
	if(using_abdomen_hough)
	{
		const fs::path abd_hough_model_name = modelname.parent_path() / modelname.stem().concat( "_abdhough" ).replace_extension(modelname.extension());
		abd_hough_forest.readFromFile(abd_hough_model_name.string(), n_trees,n_tree_levels);
		if(!abd_hough_forest.isValid())
		{
			cerr << " ERROR: invalid tree input! " << abd_hough_model_name << endl;
			return EXIT_FAILURE;
		}
		abd_hough_forest.getFeatureDefinitionString(abd_hough_feat_str);
	}

	const int n_classes = (test_problem == ut::ptClassOri) ? ori_forest.getNumberClasses() : forest.getNumberClasses() ;
	const int n_subs_classes = tracking_substructures ? subs_forest.getNumberClasses() : 0 ;

	//--------------------------------------------------------------------------
	// PARSE FOREST FEATURE SPECIFICATIONS
	//--------------------------------------------------------------------------

	// Atempt to parse the first line of the definition
	vector<ut::featType_t> feat_type;
	if(!ut::parseFeatureDefinitionString(feat_str, J, K, M, max_rot_order, feat_type, wl, feature_set_type,basis_type, Jmax, train_radius))
	{
		cerr << "ERROR: problem reading features from model file, the string was " << feat_str << endl;
		return EXIT_FAILURE;
	}

	if(any_of(feat_type.cbegin(), feat_type.cend(), [](ut::featType_t ft) { return (ft == ut::ftInvalid); }))
	{
		cerr << "ERROR: Unrecognised feature type!" << endl;
		return EXIT_FAILURE;
	}

	// Check that the substructure features match and find the Jmax values
	vector<int> subs_Jmax;
	if(tracking_substructures)
	{
		if(!checkFeaturesMatch(subs_feat_str, subs_Jmax, train_radius, J, K, M, max_rot_order, feature_set_type, basis_type, wl, feat_type))
		{
			cerr << "Error parsing the feature definition string for the substructure model, it may be invalid or may not match the other models" << endl;
			return EXIT_FAILURE;
		}
	}

	// Check that the hough features match and find the Jmax values
	vector<int> abd_hough_Jmax;
	if(using_abdomen_hough)
	{
		if(!checkHoughFeaturesMatch(abd_hough_feat_str, abd_hough_Jmax, train_radius, J, K, M, max_rot_order, feature_set_type, basis_type, wl, feat_type))
		{
			cerr << "Error parsing the feature definition string for the abdomen hough model, it may be invalid or may not match the other models" << endl;
			return EXIT_FAILURE;
		}
	}

	const int n_feat_types = feat_type.size();

	//--------------------------------------------------------------------------
	// SELECT SCALE FACTOR
	//--------------------------------------------------------------------------

	// Decide whether to use the input scale or the radius that the model was trained on (default)
	const int abd_radius = std::round(C_HEART_ABDOMEN_RATIO*radius);
	if (vm.count("scale") == 0)
	{
		scale_factor = float(train_radius)/float(radius);
		radius *= scale_factor;
		radius_int = std::ceil(radius);
	}
	else
	{
		radius *= scale_factor;
		radius_int = std::ceil(radius);
	}

	//--------------------------------------------------------------------------
	// OPEN THE VIDEO FILE
	//--------------------------------------------------------------------------

	// Open the video file
	VideoCapture vid_obj;
	if(!fs::exists(videofile))
	{
		cerr  << "ERROR: Video file does not exist" << videofile.string() << endl;
		return EXIT_FAILURE;
	}
	vid_obj.open(videofile.string());
	if ( !vid_obj.isOpened() )
	{
		cerr  << "ERROR: Could not open video file " << videofile << endl;
		return EXIT_FAILURE;
	}
	frame_rate = vid_obj.get(cv::CAP_PROP_FPS);

	if(isnan(frame_rate))
	{
		frame_rate = ut::getFrameRate(videofile.string(),videofile.parent_path().string());
		if(isnan(frame_rate))
		{
			cerr << "ERROR: Could not determine frame rate of the video " << endl;
			return EXIT_FAILURE;
		}
	}

	const int xsize = vid_obj.get(cv::CAP_PROP_FRAME_WIDTH);
	const int ysize = vid_obj.get(cv::CAP_PROP_FRAME_HEIGHT);
	int n_frames = vid_obj.get(cv::CAP_PROP_FRAME_COUNT);
	const int xresize = std::round(xsize*scale_factor);
	const int yresize = std::round(ysize*scale_factor);
	const int hough_size_x = std::round(xresize*(1.0+2.0*pad_fraction));
	const int hough_size_y = std::round(yresize*(1.0+2.0*pad_fraction));
	const int hough_offset_x = std::round(pad_fraction*xresize);
	const int hough_offset_y = std::round(pad_fraction*yresize);
	const int hough_kernel_size = hough_size_x/10 + (hough_size_x/10+1)%2;

	//--------------------------------------------------------------------------
	// READ IN THE GROUND TRUTH IF REQUIRED
	//--------------------------------------------------------------------------

	// Read in the trackfile if given
	bool gro_tru_headup;
	int gro_tru_radius, gro_tru_abd_radius;
	std::vector<bool> gro_tru_labelled, gro_tru_abd_labelled;
	std::vector<int> gro_tru_x, gro_tru_y, gro_tru_ori_degrees, gro_tru_view, gro_tru_phase_point,
					gro_tru_abd_x, gro_tru_abd_y, gro_tru_abd_ori;
	std::vector<float> gro_tru_phase;
	std::vector<ut::heartPresent_t> gro_tru_present, gro_tru_abd_present;
	std::vector<std::vector<ut::subStructLabel_t>> gro_tru_subs_track;
	if(tracking_heart && ( (display_mode == C_DISPLAY_MODE_GROUNDTRUTH) || use_gro_truth_location ) )
	{
		if(ground_truth_track_filename.string().compare("none") == 0)
		{
			cerr << "ERROR: Ground truth display (or use_ground_truth_position) requested but no track file provided (use -g option)" << endl;
			return EXIT_FAILURE;
		}

		if(!ut::readTrackFile(ground_truth_track_filename.string(), n_frames, gro_tru_headup, gro_tru_radius, gro_tru_labelled, gro_tru_present,
						   gro_tru_y, gro_tru_x, gro_tru_ori_degrees, gro_tru_view, gro_tru_phase_point, gro_tru_phase) )
		{
			cerr << "Error reading the requested track file " << ground_truth_track_filename << endl;
			return EXIT_FAILURE;
		}
	}

	if(tracking_substructures && (display_mode == C_DISPLAY_MODE_GROUNDTRUTH))
	{
		if(ground_truth_subs_filename.string() == "none")
		{
			cerr << "ERROR: Ground truth display requested but no track file provided for substructures (use -i option)" << endl;
			return EXIT_FAILURE;
		}
		vector<string> non_background_subs_names(subs_class_names.cbegin()+1,subs_class_names.cend());
		if(!readGivenSubstructures(ground_truth_subs_filename.string(), non_background_subs_names, n_frames, gro_tru_subs_track))
		{
			cerr << "Error reading the requested substructure track file " << ground_truth_subs_filename << endl;
			return EXIT_FAILURE;
		}
	}

	if(using_abdomen_hough && display_mode == C_DISPLAY_MODE_GROUNDTRUTH)
	{
		if(ground_truth_abd_filename.string().compare("none") == 0)
		{
			cerr << "ERROR: Ground truth display requested but no track file provided for abdomen (use -j option)" << endl;
			return EXIT_FAILURE;
		}
		bool gro_tru_abd_headup;
		if(!ut::readAbdomenTrackFile(ground_truth_abd_filename.string(), n_frames, gro_tru_abd_headup, gro_tru_abd_radius, gro_tru_abd_labelled, gro_tru_abd_present,
						   gro_tru_abd_y, gro_tru_abd_x, gro_tru_abd_ori) )
		{
			cerr << "Error reading the requested track file " << ground_truth_abd_filename << endl;
			return EXIT_FAILURE;
		}
	}

	//--------------------------------------------------------------------------
	// CREATE ROTATION INVARIANT FEATURE EXTRACTION OBJECTS
	//--------------------------------------------------------------------------

	// Create a feature extraction object
	vector<RIFeatures::RIFeatExtractor> feat_extractor(n_feat_types);
	for(int ft = 0; ft < n_feat_types; ++ft)
		feat_extractor[ft].initialise(cv::Size(xresize,yresize),2*radius,J[ft],K[ft],M[ft],ri_feature_method,use_memoiser,ri_coupling_method,RIFeatures::RIFeatExtractor::featureSet_enum(feature_set_type[ft]),max_rot_order[ft],RIFeatures::RIFeatExtractor::basisType_enum(basis_type[ft]));

	int max_spat_basis_halfsize = -1;
	for(int ft = 0; ft < n_feat_types; ++ft)
		max_spat_basis_halfsize = std::max(max_spat_basis_halfsize,feat_extractor[ft].getMaxSpatBasisHalfsize(Jmax[ft]));

	//--------------------------------------------------------------------------
	// READ IN MASK AND GET LISTS OF VALID PIXELS
	//--------------------------------------------------------------------------

	// Attempt to load a mask
	Mat_<unsigned char> valid_mask;
	if( !ut::prepareMask(maskstring.string(),Size(xsize,ysize),valid_mask,radius/scale_factor,Size(xresize,yresize), max_spat_basis_halfsize) )
	{
		cerr << "ERROR loading the mask: " << maskstring << endl;
		return EXIT_FAILURE;
	}

	// Get a list of valid pixels to test
	vector<cv::Point> valid_pixels;
	vector<int> img_to_valid;
	ut::findValidPixels(valid_pixels,valid_mask,1,radius_int,&img_to_valid);

	// If we are detecting substructures, we need a new mask for this
	Mat_<unsigned char> subs_valid_mask;
	vector<cv::Point> subs_valid_pixels;
	vector<int> subs_img_to_valid;
	if(tracking_substructures)
	{
		// Work out the correct radius for substructure detection based on
		// the value of Jmax
		vector<float> radius_fractions(n_feat_types);
		std::transform(subs_Jmax.cbegin(),subs_Jmax.cend(),J.cbegin(),radius_fractions.begin(),[](int jmax, int j){ return (jmax < 0) ? 1.0 : float(jmax+1)/float(j);});
		const float  subs_mask_radius_frac = *std::max_element(radius_fractions.cbegin(),radius_fractions.cend());
		radius_fractions.clear();
		int max_spat_basis_halfsize_subs = -1;
		for(int ft = 0; ft < n_feat_types; ++ft)
			max_spat_basis_halfsize_subs = std::max(max_spat_basis_halfsize_subs,feat_extractor[ft].getMaxSpatBasisHalfsize(subs_Jmax[ft]));
		if( !ut::prepareMask(maskstring.string(),Size(xsize,ysize),subs_valid_mask,std::ceil(subs_mask_radius_frac*radius/scale_factor),Size(xresize,yresize),max_spat_basis_halfsize_subs) )
		{
			cerr << "ERROR loading the mask: " << maskstring << endl;
			return EXIT_FAILURE;
		}

		// Get a list of valid pixels to test
		ut::findValidPixels(subs_valid_pixels,subs_valid_mask,1,std::ceil(radius_int*subs_mask_radius_frac),&subs_img_to_valid);
	}

	// If we are using a hough forest for the abdomen, we also need a new mask for this
	Mat_<unsigned char> hough_valid_mask;
	vector<cv::Point> hough_valid_pixels;
	if(using_abdomen_hough)
	{
		// Work out the correct radius for hough detection based on
		// the value of Jmax
		vector<float> radius_fractions(n_feat_types);
		std::transform(abd_hough_Jmax.cbegin(),abd_hough_Jmax.cend(),J.cbegin(),radius_fractions.begin(),[](int jmax, int j){ return (jmax < 0) ? 1.0 : float(jmax+1)/float(j);});
		const float abd_mask_radius_frac = *std::max_element(radius_fractions.cbegin(),radius_fractions.cend());
		radius_fractions.clear();
		int max_spat_basis_halfsize_abd_hough = -1;
		for(int ft = 0; ft < n_feat_types; ++ft)
			max_spat_basis_halfsize_abd_hough = std::max(max_spat_basis_halfsize_abd_hough,feat_extractor[ft].getMaxSpatBasisHalfsize(abd_hough_Jmax[ft]));
		if( !ut::prepareMask(maskstring.string(),Size(xsize,ysize),hough_valid_mask,std::ceil(abd_mask_radius_frac*radius/scale_factor),Size(xresize,yresize), max_spat_basis_halfsize_abd_hough) )
		{
			cerr << "ERROR loading the mask: " << maskstring << endl;
			return EXIT_FAILURE;
		}
		ut::findValidPixels(hough_valid_pixels,hough_valid_mask,pixel_skip);
	}

	//--------------------------------------------------------------------------
	// SET UP IMAGE FEATURE PROCESSORS
	//--------------------------------------------------------------------------

	const bool using_motion = any_of(feat_type.cbegin(), feat_type.cend(), [](ut::featType_t ft) { return (ft == ut::ftMotion); }) ;
	const unsigned motion_feat_index = distance(feat_type.cbegin(),find(feat_type.cbegin(), feat_type.cend(),ut::ftMotion));

	// Create objects for extracting feature representations
	vector<imageFeatureProcessor> feat_processor(n_feat_types);
	for(int ft = 0; ft < n_feat_types; ++ft)
		feat_processor[ft].initialise(feat_type[ft],yresize,xresize,wl[ft],frame_rate);

	//--------------------------------------------------------------------------
	// SET UP FOREST MODEL CALLBACKS
	//--------------------------------------------------------------------------

	auto feature_lambda = [&] (auto first_id, auto last_id, const std::array<int,2>& params, auto out_it)
	mutable
	{
		feat_extractor[params[0]].getDerivedFeature(first_id, last_id, params[1], out_it);
	};

	orientationTestingFunctor ori_functor;
	if( (test_problem == ut::ptClassOri) || (test_problem == ut::ptClassOriPhase) || (test_problem == ut::ptSubstructures) || (test_problem == ut::ptSubstructuresPCA) || (test_problem == ut::ptAbdomen) )
		ori_functor.initialise(n_feat_types,feat_extractor.data());

	// Set up the forests with the functors
	if( (test_problem == ut::ptClassOri))
	{
		ori_forest.setRegressionFunctor(&ori_functor);
	}


	if(using_joint_ori_phase_regressor)
		for(int c = 0; c < n_classes-1; ++c)
		{
			ori_phase_forest[c].setRegressionFunctor(&ori_functor);
		}

	if(test_problem == ut::ptAbdomen)
	{
		abd_hough_forest.setRegressionFunctor(&ori_functor);
	}

	// Variables needed for abdomen hough
	Mat_<float> hough_image, hough_disp;
	omp_lock_t hough_lock;

	//--------------------------------------------------------------------------
	// SET UP PARTICLE FILTERS
	//--------------------------------------------------------------------------

	// Setup particle filters or detection posteriors
	particleFilterPosClass<C_N_VIEW_CLASSES> p_filt;
	particleFilterPosClassOri<C_N_VIEW_CLASSES> p_filt_ori;
	particleFilterPosClassPhase<C_N_VIEW_CLASSES> p_filt_phase;
	particleFilterPosClassPhaseOri<C_N_VIEW_CLASSES> p_filt_ori_phase;
	particleFilterSingleStructs<C_N_VIEW_CLASSES,C_N_STRUCTURES> p_filt_subs;
	particleFilterSubStructs<C_N_VIEW_CLASSES> p_filt_subs_pca;
	vector<cp::discreteDistribution> output_posts, subs_output_posts;
	vector<cp::jointOriOutputDist<cp::circCircSingleRegressor<orientationTestingFunctor>>> ori_output_posts;
	if(use_particle_filter)
	{
		switch(test_problem)
		{
			case ut::ptClass :
				if(filter_def_files.size() != 1)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt = particleFilterPosClass<C_N_VIEW_CLASSES>(yresize, xresize, n_particles, radius, filter_def_files[0].string(), &valid_mask);
				if (!p_filt.checkInit())
				{
					cerr << "Error reading filter definition files " << endl;
					return EXIT_FAILURE;
				}
				p_filt.initialiseParticles();
				break;
			case ut::ptClassOri :
				if(filter_def_files.size() != 1)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori = particleFilterPosClassOri<C_N_VIEW_CLASSES>(yresize, xresize, n_particles, radius, filter_def_files[0].string(), &valid_mask);
				if (!p_filt_ori.checkInit())
				{
					cerr << "Error reading filter definition files " << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori.initialiseParticles();
				break;
			case ut::ptClassPhase :
				if(filter_def_files.size() != 2)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_phase = particleFilterPosClassPhase<C_N_VIEW_CLASSES>(yresize, xresize, n_particles, radius, frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), &valid_mask);
				if (!p_filt_phase.checkInit())
				{
					cerr << "Error reading filter definition file " << endl;
					return EXIT_FAILURE;
				}
				p_filt_phase.initialiseParticles();
				break;
			case ut::ptClassOriPhase :
				if(filter_def_files.size() != 2)
				{
					cerr << "Wrong number of filter definition files supplied" << endl;
					return EXIT_FAILURE;
				}
				p_filt_ori_phase = particleFilterPosClassPhaseOri<C_N_VIEW_CLASSES>(yresize, xresize, n_particles, radius, frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), &valid_mask);
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
				p_filt_subs = particleFilterSingleStructs<C_N_VIEW_CLASSES,C_N_STRUCTURES>(yresize,xresize,n_particles,radius,frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), filter_def_files[2].string(), subs_class_names, &valid_mask,&subs_valid_mask);
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
				p_filt_subs_pca = particleFilterSubStructs<C_N_VIEW_CLASSES>(yresize,xresize,n_particles,radius,frame_rate, filter_def_files[0].string(), filter_def_files[1].string(), filter_def_files[2].string(), subs_class_names, &valid_mask,&subs_valid_mask);
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
	// SET UP POSTERIORS (NON_FILTERED CASE)
	//--------------------------------------------------------------------------

	else
	// Allocate memory for the detection posteriors
	{
		if(test_problem == ut::ptClassOri)
		{
			ori_output_posts.resize(valid_pixels.size());
			for(int p = 0; p < int(valid_pixels.size()); p++)
				ori_output_posts[p].initialise(n_classes);
		}
		else
		{
			output_posts.resize(valid_pixels.size());
			for(int p = 0; p < int(valid_pixels.size()); p++)
				output_posts[p].initialise(n_classes);
		}
		if(tracking_substructures)
		{
			subs_output_posts.resize(subs_valid_pixels.size());
			for(int p = 0; p < int(subs_valid_pixels.size()); p++)
				subs_output_posts[p].initialise(n_subs_classes);
		}
	}

	vector<cp::RIHoughOutput<orientationTestingFunctor>> ri_hough_dists;
	if(using_abdomen_hough)
	{
		omp_init_lock(&hough_lock);
		ri_hough_dists.resize(hough_valid_pixels.size());
		for(unsigned p = 0; p < hough_valid_pixels.size(); ++p)
		{
			const int hx = hough_offset_x + hough_valid_pixels[p].x; // position relative to coordinates of the hough space
			const int hy = hough_offset_y + hough_valid_pixels[p].y;
			ri_hough_dists[p].initialise(hy,hx,&hough_image,hough_lock);
		}
	}

	//--------------------------------------------------------------------------
	// HELPER LAMBDAS FOR THE REWEIGHTING LAMBDAS
	//--------------------------------------------------------------------------
	const auto simple_pdf_functor = [] (const auto* node_ptr, const auto& label, const auto id)
	{
		return node_ptr->pdf(label,id);
	};

	const auto class_pdf_functor = [] (const auto* node_ptr, const auto& label, const auto /*unused*/)
	{
		return node_ptr->pdfClass(label);
	};

	const auto ori_pdf_functor = [] (const auto* node_ptr, const auto& label, const auto id)
	{
		return node_ptr->pdfOri(label,id);
	};

	const auto phase_pdf_functor = [] (const auto* node_ptr, const auto& label, const auto /*id*/)
	{
		return node_ptr->pdfPhase(label);
	};

	const auto weight_multiplier_functor = [] (const double in, const float score)
	{
		return in*score;
	};

	const auto get_point_lambda = [&] (const auto& state)
	{
		return cv::Point(std::floor(state.x),std::floor(state.y));
	};

	//--------------------------------------------------------------------------
	// SET UP CALLBACKS FOR THE PARTICLE FILTERS
	//--------------------------------------------------------------------------

	auto pos_class_reweight_lambda = [&] (auto first_point, const auto last_point, auto class_it, auto weight_it)
	mutable
	{
		forest.probabilityGroupwiseBase(first_point,last_point,class_it,weight_it,false, weight_multiplier_functor, feature_lambda, simple_pdf_functor);
	};

	auto pos_joint_class_reweight_lambda = [&] (auto first_point, const auto last_point, auto class_it, auto weight_it)
	mutable
	{
		ori_forest.probabilityGroupwiseBase(first_point,last_point,class_it,weight_it,false,weight_multiplier_functor,feature_lambda,class_pdf_functor);
	};

	auto ori_reweight_lambda = [&] (const int c, auto first_point, const auto last_point, auto ori_it, auto weight_it)
	mutable
	{
		const auto ori_pdf_per_class_functor = [=] (const auto* node_ptr, const auto& label, const auto id)
		{
			return node_ptr->pdfOri(c,label,id);
		};
		ori_forest.probabilityGroupwiseBase(first_point,last_point,ori_it,weight_it,false,weight_multiplier_functor,feature_lambda,ori_pdf_per_class_functor);
	};

	auto ori_reweight_lambda_with_phase = [&] (const int c, auto first_point, const auto last_point, auto ori_it, auto weight_it)
	mutable
	{
		ori_phase_forest[c-1].probabilityGroupwiseBase(first_point,last_point,ori_it,weight_it,false,weight_multiplier_functor,feature_lambda,ori_pdf_functor);
	};

	auto phase_reweight_lambda = [&] (const int c, auto first_pos_state, const auto last_pos_state, auto phase_it, auto weight_it)
	mutable
	{
		const auto first_point = boost::make_transform_iterator(first_pos_state,get_point_lambda);
		const auto last_point = boost::make_transform_iterator(last_pos_state,get_point_lambda);
		phase_forest[c-1].probabilityGroupwiseBase(first_point,last_point,phase_it,weight_it,false,weight_multiplier_functor,feature_lambda,simple_pdf_functor);
	};

	auto phase_reweight_lambda_with_ori = [&] (const int c, auto first_pos_state, const auto last_pos_state, auto phase_it, auto weight_it)
	mutable
	{
		const auto first_point = boost::make_transform_iterator(first_pos_state,get_point_lambda);
		const auto last_point = boost::make_transform_iterator(last_pos_state,get_point_lambda);
		ori_phase_forest[c-1].probabilityGroupwiseBase(first_point,last_point,phase_it,weight_it,false,weight_multiplier_functor,feature_lambda,phase_pdf_functor);
	};

	auto subs_reweight_lambda = [&] (auto first_point, const auto last_point, auto class_it, auto /*heart_pos_it*/, auto weight_it, const bool single_label)
	mutable
	{
		subs_forest.probabilityGroupwise(first_point,last_point,class_it,weight_it,single_label,feature_lambda);
	};

	auto pos_class_lambda_tuple = std::make_tuple(pos_class_reweight_lambda);
	auto pos_class_ori_lambda_tuple = std::make_tuple(pos_joint_class_reweight_lambda,ori_reweight_lambda);
	auto pos_class_phase_lambda_tuple = std::make_tuple(pos_class_reweight_lambda,phase_reweight_lambda);
	auto pos_class_phase_ori_lambda_tuple = std::make_tuple(pos_class_reweight_lambda,phase_reweight_lambda_with_ori,ori_reweight_lambda_with_phase);
	auto pca_subs_lambda_tuple = std::make_tuple(pos_class_reweight_lambda,phase_reweight_lambda_with_ori,ori_reweight_lambda_with_phase,subs_reweight_lambda);
	auto single_subs_lambda_tuple = std::tuple_cat( std::make_tuple(pos_class_reweight_lambda,phase_reweight_lambda_with_ori,ori_reweight_lambda_with_phase),
									ut::mp_tuple_repeat<C_N_STRUCTURES>(subs_reweight_lambda)) ;

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

	// Set up posterior images and display windows
	vector<Mat_<float>> post(n_classes-1);
	vector<Mat_<Vec3b>> post_superimposed;
	if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
	{
		if(tracking_substructures)
			post_superimposed.resize(n_subs_classes-1);
		else
			post_superimposed.resize(n_classes-1);
	}
	for(int c = 1; c < n_classes; ++c)
	{
		if((display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE) && (!tracking_substructures) && (tracking_heart))
		{
			namedWindow("Detection Posterior " + class_names[c], WINDOW_AUTOSIZE );
			moveWindow("Detection Posterior " + class_names[c], (c/2)*xsize,(c%2)*ysize);
		}
		post[c-1] = Mat::zeros(yresize,xresize, CV_32F);
	}

	// Posteriors for substructures
	vector<Mat_<float>> subs_post;
	if(tracking_substructures)
	{
		subs_post.resize(n_subs_classes-1);
		for(int c = 1; c < n_subs_classes; ++c)
		{
			if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
			{
				namedWindow("Detection Posterior " + subs_class_names[c], WINDOW_AUTOSIZE );
				moveWindow("Detection Posterior " + subs_class_names[c], (c/2)*xsize,(c%2)*ysize);
			}
			subs_post[c-1] = Mat::zeros(yresize,xresize, CV_32F);
		}
	}

	// Set up for display of hough image
	if(!tracking_heart && ((display_mode == C_DISPLAY_MODE_ALL) || (display_mode == C_DISPLAY_MODE_NO_IMPOSE)))
	{
		namedWindow("Hough Image",WINDOW_AUTOSIZE);
	}

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
	// SET UP RECORDING
	//--------------------------------------------------------------------------

	// Set up a video to record
	Size record_vid_size;
	if(record)
	{
		switch(display_mode)
		{
			case C_DISPLAY_MODE_NONE:
				cerr << "ERROR: Cannot use record mode with no display" << endl;
				return EXIT_FAILURE;
			case C_DISPLAY_MODE_DETECTION:
				record_vid_size = Size(xsize,ysize);
				break;
			case C_DISPLAY_MODE_ALL:
			case C_DISPLAY_MODE_NO_IMPOSE:
				record_vid_size = Size(2*xsize,2*ysize);
				break;
			case C_DISPLAY_MODE_GROUNDTRUTH:
				record_vid_size = Size(xsize,2*ysize);
				break;
		}

		int ex = static_cast<int>(vid_obj.get(cv::CAP_PROP_FOURCC));
		output_video.open(record_vidname.string(), ex, frame_rate, record_vid_size, true);

		if (!output_video.isOpened())
		{
			cerr  << "Could not open the output video for write: " << record_vidname << endl;
			return -1;
		}
	}

	// Set up parameters of the posterior filtering
	//int filt_size = std::round(C_GAUSS_BLUR_SIZE*scale_factor);
	//if(filt_size % 2 == 0)
	//	filt_size++;

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
				output_file << "weight abdomen_y abdomen_x\n";
				break;
		}
	}

	//--------------------------------------------------------------------------
	// START THE LOOP
	//--------------------------------------------------------------------------

	double frame_start = clock();

	// Loop through frames
	const double test_start = (double) getTickCount();
	for(int f = 0 ; f < n_frames; ++f)
	{
		//--------------------------------------------------------------------------
		// LOOP VARIABLES
		//--------------------------------------------------------------------------
		Mat_<unsigned char> I, I_resize;
		Mat_<Vec3b> disp, gro_tru_disp;
		double overall_max_val, abdomen_max_val, ori = 0.0;
		bool heart_visible = true;
		int overall_max_point_x, overall_max_point_y;
		int abdomen_max_point_x, abdomen_max_point_y;
		float max_phase = 0.0, heart_rate = 0.0;

		//--------------------------------------------------------------------------
		// TIMINGS
		//--------------------------------------------------------------------------

		// Timers to time since the start of the previous loop
		// in order to monitor frame rate
		const double frame_stop = (double) getTickCount();
		const double achieved_fps = getTickFrequency()/double(frame_stop - frame_start);
		frame_start = (double) getTickCount();

		//--------------------------------------------------------------------------
		// READ IN FRAME
		//--------------------------------------------------------------------------

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

		//--------------------------------------------------------------------------
		// PROCESS FRAME AND PUT IN FEATURE EXTRACTOR
		//--------------------------------------------------------------------------

		// Transform the image to the desired representation
		// and place into the feature extractor
		for(int ft = 0; ft < n_feat_types; ++ft)
		{
			Mat_<float> Imag, Iori;
			feat_processor[ft].extract(I_resize,Imag,Iori);
			if(feat_type[ft] == ut::ftInt)
				feat_extractor[ft].setScalarInputImage(Imag);
			else
				feat_extractor[ft].setVectorInputImage(Imag,Iori);
		}

		// If we are using motion features, we can't do anything more on the first frame
		if( (f == 0) && using_motion)
			continue;

		if(using_abdomen_hough)
		{
			// Use the Hough forest for prediction
			hough_image = Mat_<float>::zeros(hough_size_y, hough_size_x);
			abd_hough_forest.predictDistGroupwise(hough_valid_pixels.cbegin(), hough_valid_pixels.cend(), ri_hough_dists.begin(), feature_lambda);
			GaussianBlur(hough_image,hough_image,Size(hough_kernel_size,hough_kernel_size),hough_kernel_size/6.0,hough_kernel_size/6.0,BORDER_CONSTANT);
		}

		//--------------------------------------------------------------------------
		// DETECTION IN THE NON-FILTERING CASE
		//--------------------------------------------------------------------------

		// Use random forests for detection
		if(!use_particle_filter)
		{
			if(test_problem == ut::ptClassOri)
			{
				ori_forest.predictDistGroupwise(valid_pixels.cbegin(), valid_pixels.cend(),ori_output_posts.begin(), feature_lambda);

				for(int p = 0; p < int(valid_pixels.size()); ++p)
				{
					for(int c = 1; c < n_classes; ++c)
						post[c-1](valid_pixels[p]) = ori_output_posts[p].d_dist.pdf(c);
				}
			}
			else if (tracking_heart)
			{
				forest.predictDistGroupwise(valid_pixels.cbegin(), valid_pixels.cend(), output_posts.begin(), feature_lambda);

				for(int p = 0; p < int(valid_pixels.size()); p++)
				{
					for(int c = 1; c < n_classes; ++c)
						post[c-1](valid_pixels[p]) = output_posts[p].pdf(c);
				}
			}

			if(test_problem == ut::ptAbdomen)
			{
				// Find the most likely point in the abdomen hough image
				double min_val, max_val;
				Point min_point, max_point;
				Rect window_rect;
				minMaxLoc(hough_image,&min_val,&max_val,&min_point,&max_point);
				window_rect = Rect(max_point,Size(15,15));
				const TermCriteria mean_shift_term_crit(1,100,1.0);
				meanShift(hough_image,window_rect,mean_shift_term_crit);
				const double max_x_resize = (window_rect.x+window_rect.width/2.0);
				const double max_y_resize = (window_rect.y+window_rect.height/2.0);
				abdomen_max_val = max_val;
				abdomen_max_point_x = (max_x_resize - hough_offset_x)/scale_factor;
				abdomen_max_point_y = (max_y_resize - hough_offset_y)/scale_factor;
				if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
					normalize(hough_image,hough_disp,0,1, cv::NORM_MINMAX);
			}
			else
			{
				cv::Point max_point_resize;
				overall_max_val = -1.0;
				#pragma omp parallel for
				for(int c = 1; c < n_classes; ++c)
				{
					double min_val, max_val;
					Point min_point, max_point;
					Rect window_rect;

					//GaussianBlur(post[c-1],post[c-1],Size(filt_size,filt_size),filt_size/3.0,filt_size/3.0,BORDER_CONSTANT);
					minMaxLoc(post[c-1],&min_val,&max_val,&min_point,&max_point);
					window_rect = Rect(max_point,mean_shift_size);
					meanShift(post[c-1],window_rect,mean_shift_term_crit);
					#pragma omp critical
					{
						if(max_val > overall_max_val)
						{
							overall_max_val = max_val;
							max_point_resize.x = (window_rect.x+window_rect.width/2.0);
							max_point_resize.y = (window_rect.y+window_rect.height/2.0);
							max_class = c;
						}
					}

					// Superimpose the detection posterior over the input image
					if( (display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE) && (!tracking_substructures) )
					{
						Mat colour_channels[3];
						colour_channels[0] = Mat::zeros(post[c-1].rows,post[c-1].cols,CV_8U);
						colour_channels[1] = Mat::zeros(post[c-1].rows,post[c-1].cols,CV_8U);
						colour_channels[2] = 255*post[c-1];
						colour_channels[2].convertTo(colour_channels[2],CV_8U);
						merge(colour_channels,3,post_superimposed[c-1]);
						resize(post_superimposed[c-1],post_superimposed[c-1],Size(xsize,ysize));
						if(display_mode == C_DISPLAY_MODE_ALL)
							post_superimposed[c-1] = post_superimposed[c-1] + disp;
					}
				}
				overall_max_point_x = max_point_resize.x/scale_factor;
				overall_max_point_y = max_point_resize.y/scale_factor;

				// Find the predicted orientation if needed (first check that it's a valid position)
				if(max_point_resize.x <= radius_int) max_point_resize.x = radius_int + 1;
				if(max_point_resize.x >= xresize-radius_int) max_point_resize.x = xresize - radius_int - 1;
				if(max_point_resize.y <= radius_int) max_point_resize.y = radius_int + 1;
				if(max_point_resize.y >= yresize-radius_int) max_point_resize.y = yresize - radius_int - 1;
				const int maxid = max_point_resize.y*xresize + max_point_resize.x;

				int gro_tru_valid_id = 0;
				cv::Point gro_tru_point_resize;
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

				// Get the orientation and/or phase at the point
				switch(test_problem)
				{
					case ut::ptAbdomen:
					case ut::ptClass :
						// Nothing more to do
						break;
					case ut::ptClassOri :
						if(use_gro_truth_location)
						{
							// Orientation at the true position
							if(gro_tru_present[f] > 0)
								ori = ori_output_posts[gro_tru_valid_id].vm_dist[gro_tru_view[f]-1].getMu();
							else
							{
								ori = 0.0;
								break;
							}
						}
						else
							// Orientation at the predicted position
							ori = ori_output_posts[img_to_valid[maxid]].vm_dist[max_class-1].getMu();

						break;
					case ut::ptClassPhase :
						{
							cp::vonMisesDistribution phase_dist;
							if(use_gro_truth_location)
							{
								if(gro_tru_present[f] > 0)
									phase_forest[gro_tru_view[f]-1].predictDistGroupwise(&gro_tru_point_resize,(&gro_tru_point_resize)+1,&phase_dist, feature_lambda);
								else
								{
									max_phase = 0.0;
									break;
								}
							}
							else
								phase_forest[max_class-1].predictDistGroupwise(&max_point_resize,(&max_point_resize)+1,&phase_dist, feature_lambda);
							max_phase = phase_dist.getMu();
						}
						break;
					case ut::ptClassOriPhase :
					case ut::ptSubstructures :
					case ut::ptSubstructuresPCA :
						{
							cp::jointOriPhaseOutputDist<cp::circCircSingleRegressor<orientationTestingFunctor>> ori_phase_dist;
							if(use_gro_truth_location)
							{
								if(gro_tru_present[f] > 0)
									ori_phase_forest[gro_tru_view[f]-1].predictDistGroupwise(&gro_tru_point_resize,(&gro_tru_point_resize)+1,&ori_phase_dist,feature_lambda);
								else
								{
									ori = 0.0;
									max_phase = 0.0;
									break;
								}
							}
							else
								ori_phase_forest[max_class-1].predictDistGroupwise(&max_point_resize,(&max_point_resize)+1,&ori_phase_dist, feature_lambda);

							ori = ori_phase_dist.vm_dist_angle.getMu();
							max_phase = ori_phase_dist.vm_dist_phase.getMu();
						}
						break;
				}
			}

			// Now detect the substructure positions, if required
			// Substructure detections
			if(tracking_substructures)
			{
				subs_forest.predictDistGroupwise(subs_valid_pixels.cbegin(), subs_valid_pixels.cend(), subs_output_posts.begin(), feature_lambda);

				for(int p = 0; p < int(subs_valid_pixels.size()); ++p)
				{
					for(int c = 1; c < n_subs_classes; ++c)
						subs_post[c-1](subs_valid_pixels[p]) = subs_output_posts[p].pdf(c);
				}

				#pragma omp parallel for
				for(int c = 1; c < n_subs_classes; ++c)
				{
					double min_val, max_val;
					Point min_point, max_point;
					Rect window_rect;
					Mat colour_channels[3];

					//GaussianBlur(post[c-1],post[c-1],Size(filtsize,filtsize),filtsize/3.0,filtsize/3.0,BORDER_CONSTANT);
					minMaxLoc(subs_post[c-1],&min_val,&max_val,&min_point,&max_point);
					window_rect = Rect(max_point,mean_shift_size);
					meanShift(subs_post[c-1],window_rect,mean_shift_term_crit);
					const int max_subs_x_resize = (window_rect.x+window_rect.width/2.0);
					const int max_subs_y_resize = (window_rect.y+window_rect.height/2.0);
					subs_locations[c-1] = Point(max_subs_x_resize/scale_factor,max_subs_y_resize/scale_factor);

					// Superimpose the detection posterior over the input image
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
					{
						colour_channels[0] = Mat::zeros(subs_post[c-1].rows,subs_post[c-1].cols,CV_8U);
						colour_channels[1] = Mat::zeros(subs_post[c-1].rows,subs_post[c-1].cols,CV_8U);
						colour_channels[2] = 255*subs_post[c-1];
						colour_channels[2].convertTo(colour_channels[2],CV_8U);
						merge(colour_channels,3,post_superimposed[c-1]);
						resize(post_superimposed[c-1],post_superimposed[c-1],Size(xsize,ysize));
						if(display_mode == C_DISPLAY_MODE_ALL)
							post_superimposed[c-1] = post_superimposed[c-1] + disp;
					}
				}
			}
		}
		//--------------------------------------------------------------------------
		// UPDATE PARTICLE FILTERS
		//--------------------------------------------------------------------------
		else
			// Use some form of particle filter
		{
			// Prepare the display images for the visualisation functions
			if(display_mode == C_DISPLAY_MODE_ALL )
			{
				for(auto& post : post_superimposed)
					disp.copyTo(post);
			}
			else if (display_mode == C_DISPLAY_MODE_NO_IMPOSE)
			{
				for(auto& post : post_superimposed)
					post = Mat_<Vec3b>::zeros(ysize,xsize);
			}

			switch(test_problem)
			{
				case ut::ptClass :
				{
					if(using_motion)
						p_filt.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pos_class_lambda_tuple);
					else
						p_filt.step(I_resize,pos_class_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
						p_filt.visualiseHidden(post_superimposed.data());
					std::tuple<statePosClass> class_pos_state;
					p_filt.meanShiftEstimate(class_pos_state,overall_max_val);
					max_class = std::get<0>(class_pos_state).c;
					heart_visible = std::get<0>(class_pos_state).visible;
					overall_max_point_x = int(std::get<0>(class_pos_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(class_pos_state).y/scale_factor);
				}
				break;

				case ut::ptClassOri :
				{
					if(using_motion)
						p_filt_ori.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pos_class_ori_lambda_tuple);
					else
						p_filt_ori.step(I_resize,pos_class_ori_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
						p_filt_ori.visualiseOri(post_superimposed.data());
					std::tuple<statePosClass,stateOri> class_pos_ori_state;
					p_filt_ori.meanShiftEstimate(class_pos_ori_state,overall_max_val);
					max_class = std::get<0>(class_pos_ori_state).c;
					ori = std::get<1>(class_pos_ori_state).ori;
					heart_visible = std::get<0>(class_pos_ori_state).visible;
					overall_max_point_x = int(std::get<0>(class_pos_ori_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(class_pos_ori_state).y/scale_factor);
				}
				break;

				case ut::ptClassPhase :
				{
					if(using_motion)
						p_filt_phase.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pos_class_phase_lambda_tuple);
					else
						p_filt_phase.step(I_resize,pos_class_phase_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
						p_filt_phase.visualisePhase(post_superimposed.data());
					std::tuple<statePosClass,statePhase> class_pos_phase_state;
					p_filt_phase.meanShiftEstimate(class_pos_phase_state,overall_max_val);
					max_class = std::get<0>(class_pos_phase_state).c;
					max_phase = std::get<1>(class_pos_phase_state).ph;
					heart_rate = std::get<1>(class_pos_phase_state).ph_rate*(60.0/(2.0*M_PI));
					heart_visible = std::get<0>(class_pos_phase_state).visible;
					overall_max_point_x = int(std::get<0>(class_pos_phase_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(class_pos_phase_state).y/scale_factor);
				}
				break;

				case ut::ptClassOriPhase :
				{

					if(using_motion)
						p_filt_ori_phase.step(feat_processor[motion_feat_index].getUnnormalisedFlow(),pos_class_phase_ori_lambda_tuple);
					else
						p_filt_ori_phase.step(I_resize,pos_class_phase_ori_lambda_tuple);
					if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
						p_filt_ori_phase.visualiseOriPhase(post_superimposed.data());
					std::tuple<statePosClass,statePhase,stateOri> class_pos_phase_ori_state;
					p_filt_ori_phase.meanShiftEstimate(class_pos_phase_ori_state,overall_max_val);
					max_class = std::get<0>(class_pos_phase_ori_state).c;
					max_phase = std::get<1>(class_pos_phase_ori_state).ph;
					heart_rate = std::get<1>(class_pos_phase_ori_state).ph_rate*(60.0/(2.0*M_PI));
					ori = std::get<2>(class_pos_phase_ori_state).ori;
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
						p_filt_subs.visualiseSubstructures(post_superimposed.data());
					typename particleFilterSingleStructs<C_N_VIEW_CLASSES,C_N_STRUCTURES>::combined_state_type subs_state;
					p_filt_subs.meanShiftEstimate(subs_state,overall_max_val);
					max_class = std::get<0>(subs_state).c;
					max_phase = std::get<1>(subs_state).ph;
					heart_rate = std::get<1>(subs_state).ph_rate*(60.0/(2.0*M_PI));
					ori = std::get<2>(subs_state).ori;
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
						p_filt_subs_pca.visualiseSubstructures(post_superimposed.data());
					std::tuple<statePosClass,statePhase,stateOri,stateSubstructuresPCA<C_N_VIEW_CLASSES>> subs_state;
					p_filt_subs_pca.meanShiftEstimate(subs_state,overall_max_val);
					max_class = std::get<0>(subs_state).c;
					max_phase = std::get<1>(subs_state).ph;
					heart_rate = std::get<1>(subs_state).ph_rate*(60.0/(2.0*M_PI));
					ori = std::get<2>(subs_state).ori;
					heart_visible = std::get<0>(subs_state).visible;
					overall_max_point_x = int(std::get<0>(subs_state).x/scale_factor);
					overall_max_point_y = int(std::get<0>(subs_state).y/scale_factor);
					for(int s = 0; s < n_subs_classes-1; ++s)
					{
						subs_locations[s] = Point(std::get<3>(subs_state).x[s]/scale_factor,std::get<3>(subs_state).y[s]/scale_factor);
						subs_visible_arr[s] = std::get<3>(subs_state).visible[s];
					}

				}
				break;
			}
		}

		//--------------------------------------------------------------------------
		// OUTPUT DETECTIONS
		//--------------------------------------------------------------------------

		// Output the detections if desired
		if(output_detections)
		{
			output_file << f << " " ;

			if(test_problem == ut::ptAbdomen)
				output_file << abdomen_max_val << " " << abdomen_max_point_y << " " << abdomen_max_point_x;

			if(tracking_heart)
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

		//--------------------------------------------------------------------------
		// DISPLAY THIS FRAME
		//--------------------------------------------------------------------------

		// Display the ground truth if desired
		if(display_mode == C_DISPLAY_MODE_GROUNDTRUTH)
		{
			gro_tru_disp = disp.clone();


			// Draw circle for position of the detection
			if(tracking_heart && gro_tru_labelled[f] && (gro_tru_present[f] != ut::hpNone) )
				displayHeart(gro_tru_disp, gro_tru_x[f],gro_tru_y[f], gro_tru_view[f], float(gro_tru_ori_degrees[f])*M_PI/180.0, gro_tru_phase[f], gro_tru_present[f], test_problem, radius/scale_factor);

			if(using_abdomen_hough && gro_tru_abd_labelled[f] && (gro_tru_abd_present[f] != ut::hpNone) )
			{
				const int line_thickness = 2;
				const Point gro_tru_abd_point(gro_tru_abd_x[f],gro_tru_abd_y[f]);
				circle(gro_tru_disp,gro_tru_abd_point,gro_tru_abd_radius,CLR_ABDOMEN,line_thickness);
			}

			// Display the substructure detections
			if(tracking_substructures)
			{
				for(int s = 1; s < n_subs_classes; ++s)
				{
					if(gro_tru_subs_track[f][s-1].labelled && (gro_tru_subs_track[f][s-1].present == 1)
						&& ((test_problem == ut::ptSubstructuresPCA && p_filt_subs_pca.structInView(s-1,gro_tru_view[f])) || (test_problem == ut::ptSubstructures && p_filt_subs.structInView(s-1,gro_tru_view[f]))) )
					{
						const Point subs_loc(gro_tru_subs_track[f][s-1].x,gro_tru_subs_track[f][s-1].y);
						circle(gro_tru_disp,subs_loc,3,Scalar(0,0,255),-1);
					}
				}
			}

			// Display the main image
			imshow("Ground Truth", gro_tru_disp);
		}

		// Display the main detection window
		if(display_mode != C_DISPLAY_MODE_NONE)
		{
			if(!pause)
				putText(disp,string("FPS ") + to_string(achieved_fps),Point(5,10),FONT_HERSHEY_PLAIN,1.0,Scalar(0,127,127) );

			if( (tracking_heart) && (overall_max_val >= detection_threshold) && (heart_visible || display_hidden) )
			{
				displayHeart(disp, overall_max_point_x, overall_max_point_y, max_class, ori, max_phase, heart_visible ? ut::hpPresent : ut::hpNone, test_problem, radius/scale_factor);

				if(use_particle_filter && !record && (test_problem == ut::ptClassPhase || test_problem == ut::ptClassOriPhase  || test_problem == ut::ptSubstructures || test_problem == ut::ptSubstructuresPCA))
					putText(disp,string("BPM ") + to_string(int(heart_rate)),Point(overall_max_point_x + (0.5*radius/scale_factor)*std::cos(ori), overall_max_point_y - (1.3*radius/scale_factor)*std::sin(-ori) ),FONT_HERSHEY_PLAIN,1.0,Scalar(0,127,127) );
			}

			if(using_abdomen_hough)
				circle(disp,Point(abdomen_max_point_x,abdomen_max_point_y),abd_radius,CLR_ABDOMEN,2);

			// Display the substructure detections
			if(tracking_substructures)
			{
				const cv::Scalar subs_colour = heart_visible ? CLR_SUBSTRUCTURES : CLR_HIDDEN;
				for(int s = 1; s < n_subs_classes; ++s)
				{
					if(subs_visible_arr[s-1] == structVisible_enum::svVisible)
						circle(disp,subs_locations[s-1],3,subs_colour,-1);
				}
			}

			// Display the main image
			imshow("Input Stream", disp);
		}


		if(display_mode == C_DISPLAY_MODE_ALL || display_mode == C_DISPLAY_MODE_NO_IMPOSE)
		{
			if(tracking_substructures)
				for(int c = 1; c < n_subs_classes; ++c)
					imshow("Detection Posterior " + subs_class_names[c],post_superimposed[c-1]);
			else if (test_problem == ut::ptAbdomen)
					imshow("Hough Image", hough_disp);
			else
				for(int c = 1; c < n_classes; ++c)
					imshow("Detection Posterior " + class_names[c],post_superimposed[c-1]);
		}

		//--------------------------------------------------------------------------
		// RECORD THIS FRAME
		//--------------------------------------------------------------------------

		if(record)
		{
			switch(display_mode)
			{
				case C_DISPLAY_MODE_DETECTION:
					output_video << disp;
					break;

				case C_DISPLAY_MODE_ALL:
				case C_DISPLAY_MODE_NO_IMPOSE:
					{
						Mat_<Vec3b> record_frame = Mat_<Vec3b>::zeros(2*ysize,2*xsize);
						Mat_<Vec3b> roi = Mat(record_frame,Rect(0,0,xsize,ysize));
						disp.copyTo(roi);
						roi = Mat(record_frame,Rect(0,ysize,xsize,ysize));
						post_superimposed[0].copyTo(roi);
						roi = Mat(record_frame,Rect(xsize,0,xsize,ysize));
						post_superimposed[1].copyTo(roi);
						roi = Mat(record_frame,Rect(xsize,ysize,xsize,ysize));
						post_superimposed[2].copyTo(roi);
						output_video << record_frame;
					}
					break;

				case C_DISPLAY_MODE_GROUNDTRUTH:
					{
						Mat_<Vec3b> record_frame = Mat_<Vec3b>::zeros(2*ysize,xsize);
						Mat_<Vec3b> roi = Mat(record_frame,Rect(0,0,xsize,ysize));
						disp.copyTo(roi);
						roi = Mat(record_frame,Rect(0,ysize,xsize,ysize));
						gro_tru_disp.copyTo(roi);
						output_video << record_frame;
					}
					break;

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

	if(record)
		output_video.release();

	// Tidy up
	if(using_abdomen_hough)
		omp_destroy_lock(&hough_lock);
	vid_obj.release();

}
