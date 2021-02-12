#include <iostream>
#include <string>
#include <omp.h>
#include <map>
#include <cassert>
#include <random>
#include <algorithm> /* std::copy, std::transform */
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <RIFeatures/RIFeatExtractor.hpp>
#include <canopy/classifier/classifier.hpp>
#include "jointOrientationRegressor.hpp"
#include "jointOriPhaseRegressor.hpp"
#include <canopy/circularRegressor/circularRegressor.hpp>
#include "circCircSingleRegressor.hpp"
#include "thesisUtilities.h"
#include "displayFunctions.h"
#include "imageFeatureProcessor.h"
#include "trainingFunctors.h"

// Main entry point for training application
int main( int argc, char** argv )
{
	// Namespaces
	using namespace cv;
	using namespace std;
	namespace po = boost::program_options;
	namespace fs = boost::filesystem;
	namespace ut = thesisUtilities;
	using namespace RIFeatures;
	namespace cp = canopy;

	// Default parameters
	const std::string C_DEFAULT_FEAT = "int";
	constexpr int C_DEFAULT_J = 5;
	constexpr int C_DEFAULT_K = 4;
	constexpr int C_DEFAULT_M = 0;
	constexpr int C_DEFAULT_TREES = 8;
	constexpr int C_DEFAULT_LEVELS = 10;
	constexpr double C_DEFAULT_WAVELENGTH = 0.0;
	constexpr int C_DEFAULT_RADIUS = 30;
	constexpr RIFeatExtractor::featureSet_enum C_DEFAULT_FEATURE_SET_TYPE = RIFeatExtractor::fsSimpleCouple;

	// View Colours - for display purposes only
	const cv::Scalar CLR_4CHAM(255,0,0); // blue
	const cv::Scalar CLR_LVOT(0,255,0); // green
	const cv::Scalar CLR_RVOT(0,255,255); // yellow
	const cv::Scalar CLR_VSIGN(0,0,255);  // red

	// View Codes
	constexpr int VIEW_4CHAM = 1;
	constexpr int VIEW_LVOT = 2;
	constexpr int VIEW_RVOT = 3;
	constexpr int VIEW_VSIGN = 4;

	// Declarations
	vector<string> uniquevidname;
	vector<int> frameno;
	vector<int> label, centrex, centrey, vidindex;
	vector<float> vidradius, orientation, cardiacphase;
	vector<string> class_names;
	vector<vector<int>> datapoints_per_vid, ids_per_label;
	bool read_error = false;
	int train_radius;
	int previousvid = -1, n_class_labels;
	fs::path datasetfile, viddir, modelfilename;
	string feature_header, feat_str;
	stringstream feat_stream;

	// Forest parameters
	int num_trees, tree_depth, num_training_features;

	// Rotation Invariant Description Parameters
	vector<string> feat_type_string;
	vector<string> feature_set_string;
	vector<RIFeatExtractor::featureSet_enum> feature_set_type;
	vector<int> J, K, M, Jmax;
	vector<float> wl;
	// ***EDIT THESE***
	int max_rot_order = -1;
	constexpr RIFeatExtractor::basisType_enum basis_type = RIFeatExtractor::btSoftHist;

	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("dataset,d", po::value<fs::path>(&datasetfile), "input dataset file")
		("videodirectory,v", po::value<fs::path>(&viddir)->default_value("."), "directory containing video files")
		("modelfile,o", po::value<fs::path>(&modelfilename)->default_value("modelout"), "root name for output files")
		("featureType,f",po::value<vector<string>>(&feat_type_string)->multitoken(), "type of feature used (multiple arguments may be listed")
		("num_training_features,t",po::value<int>(&num_training_features)->default_value(-1), "number of features to consider at each split node")
		("nj,j",po::value<vector<int>>(&J)->multitoken(), "number of radial bases")
		("nk,k",po::value<vector<int>>(&K)->multitoken(), "max basis rotation order")
		("nm,m",po::value<vector<int>>(&M)->multitoken(), "Fourier histogram expansion order")
		("feature_set,c",po::value<vector<string>>(&feature_set_string)->multitoken(), "feature coupling type (basic/couple_simple/couple_extra)")
		("jmax,x",po::value<vector<int>>(&Jmax)->multitoken(), "maximum value of j that is permitted to be used")
		("radius,r",po::value<int>(&train_radius)->default_value(C_DEFAULT_RADIUS), "radius at which to detect structures in the trained model")
		("wavelength,w",po::value<vector<float>>(&wl)->multitoken(), "monogenic filter centre-wavelength")
		("trees,n",po::value<int>(&num_trees)->default_value(C_DEFAULT_TREES), "number of trees")
		("depth,l",po::value<int>(&tree_depth)->default_value(C_DEFAULT_LEVELS), "number of levels per tree")
		("without_split_node_dists,s" , "do not fit node distributions to split nodes in the forest")
		("orientation,a" , "also train an orientation prediction model")
		("phase,p" , "train an additional phase prediction model")
		("display,D" , "display training examples");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		cout << "Train a random forest using rotation invariant features" << endl;
		cout << desc << endl;
		return EXIT_SUCCESS;
	}

	if (vm.count("dataset") != 1)
	{
		cerr << "No dataset file specified (use -d option)" << endl;
		return EXIT_FAILURE;
	}

	const bool fit_split_dists = !bool(vm.count("without_split_node_dists"));

	// Set booleans according to options used
	const bool display_mode = (vm.count("display") > 0);
	const ut::problemType_t train_problem = ut::boolsToProblemType((vm.count("orientation") > 0),(vm.count("phase") > 0));

	// Interpret the feature type strings
	const int num_feat_types = feat_type_string.size();
	if(num_feat_types == 0)
	{
		cerr << "No feature type specified" << endl;
		return EXIT_FAILURE;
	}
	vector<ut::featType_t> feat_type;
	feat_type.resize(num_feat_types);
	transform(feat_type_string.cbegin(),feat_type_string.cend(),feat_type.begin(),ut::strToFeat);

	for(auto ft : feat_type)
	{
		if(ut::ftInvalid == ft)
		{
			cerr << "Invalid feature type argument" << endl;
			return EXIT_FAILURE;
		}
	}

	// Fill in default arguments where required
	if(!ut::pad_vector_args<int>(J,num_feat_types,C_DEFAULT_J))
	{
		cerr << "Unexpected number of J parameters" << endl;
		return EXIT_FAILURE;
	}
	if(!ut::pad_vector_args<int>(K,num_feat_types,C_DEFAULT_K))
	{
		cerr << "Unexpected number of K parameters" << endl;
		return EXIT_FAILURE;
	}
	if(!ut::pad_vector_args<int>(M,num_feat_types,C_DEFAULT_M))
	{
		cerr << "Unexpected number of M parameters" << endl;
		return EXIT_FAILURE;
	}
	if(!ut::pad_vector_args<float>(wl,num_feat_types,C_DEFAULT_WAVELENGTH))
	{
		cerr << "Unexpected number of wavelength parameters" << endl;
		return EXIT_FAILURE;
	}
	if(!ut::pad_vector_args<int>(Jmax,num_feat_types,-1))
	{
		cerr << "Unexpected number of Jmax parameters" << endl;
		return EXIT_FAILURE;
	}

	// Interpret the feature set strings
	vector<bool> use_memoiser;
	for(auto ct_string : feature_set_string)
	{
		RIFeatExtractor::featureSet_enum fs;
		if(!RIFeatExtractor::stringToFeatureSet(ct_string,fs))
		{
			cerr << "Unrecognised feature set string " << ct_string << endl;
			return EXIT_FAILURE;
		}
		feature_set_type.emplace_back(fs);
	}
	if(!ut::pad_vector_args<RIFeatExtractor::featureSet_enum>(feature_set_type,num_feat_types,C_DEFAULT_FEATURE_SET_TYPE))
	{
		cerr << "Unexpected number of feature_set parameters" << endl;
		return EXIT_FAILURE;
	}
	for(RIFeatExtractor::featureSet_enum fs : feature_set_type)
		use_memoiser.push_back((fs != RIFeatExtractor::fsBasic)); // use memoiser for feature sets with coupling


	// Check for sensible input values
	for(int ft = 0; ft < num_feat_types; ++ft)
	{
		if( (J[ft] < 1) || (K[ft] < 0) || (M[ft] < 0) || ( (feat_type[ft] == ut::ftInt) && (M[ft] != 0) ) || ( (feat_type[ft] != ut::ftInt) && (M[ft] < 1) ))
		{
			cerr << "Invalid combination of J,K,M for feature " << ft <<  endl;
			return EXIT_FAILURE;
		}

		if (wl[ft] < 0.0)
		{
			cerr << "Must use a positive wavelength" << endl;
			return EXIT_FAILURE;
		}

		if (Jmax[ft] > J[ft])
		{
			cerr << "Jmax must be <= J for feature " << ft << endl;
			return EXIT_FAILURE;
		}
	}

	/*if( (train_problem == ut::ptClassOri || train_problem == ut::ptClassOriPhase) && any_of(Jmax.cbegin(),Jmax.cend(),[](int i) { return (i > 0); }) )
	{
		cerr << "Use of maximum J values with orientation features is not yet implemented" << endl;
		return EXIT_FAILURE;
	}*/

	if( (num_trees < 1) || (tree_depth < 1) )
	{
		cerr << "Invalid tree parameters" << endl;
		return EXIT_FAILURE;
	}

	// Read the dataset file into arrays
	if(!ut::readDataset(datasetfile.string(),class_names,uniquevidname,datapoints_per_vid,frameno,vidradius,label,orientation,vidindex,centrey,centrex,cardiacphase))
	{
		cerr << "Error reading file '" << datasetfile << "'." << endl;
		return EXIT_FAILURE;
	}
	const int n_data = vidindex.size();
	const int n_vids = uniquevidname.size();
	n_class_labels = 1;

	for(int d = 0; d < n_data; d++)
	{
		if(label[d] > n_class_labels - 1)
			n_class_labels = label[d] + 1;
	}

	// Sort out the data points into vectors of different view classes
	if( (train_problem == ut::ptClassPhase) || (train_problem == ut::ptClassOriPhase) )
	{
		ids_per_label.resize(n_class_labels-1);
		for(int d = 0; d < n_data; ++d)
			if(label[d] > 0)
				ids_per_label[label[d]-1].emplace_back(d);
	}

	// Create arrays to hold the precomputed feature values
	vector<vector<vector<float>>> sinorifeatsarray, cosorifeatsarray;
	vector<vector<vector<float>>> featarray(num_feat_types,vector<vector<float>>(n_data));
	vector<cp::classOriLabel_t> ori_labels;
	vector<cp::phaseOriLabel_t> ori_phase_labels;
	vector<int> num_ori_feats_per_type;
	vector<vector<int>> ori_feats_lists;

	if( (train_problem == ut::ptClassOri) || (train_problem == ut::ptClassOriPhase) )
	{
		sinorifeatsarray.resize(num_feat_types,vector<vector<float>>(n_data));
		cosorifeatsarray.resize(num_feat_types,vector<vector<float>>(n_data));

		if(train_problem == ut::ptClassOri)
			ori_labels.resize(n_data);
		else if(train_problem == ut::ptClassOriPhase)
			ori_phase_labels.resize(n_data);

		num_ori_feats_per_type.resize(num_feat_types);
		ori_feats_lists.resize(num_feat_types);
	}

	// A dummy feature extractor to get the lists of orientation features of order one for this set of parameters
	std::vector<std::vector<int>> low_j_feats(num_feat_types);
	vector<int> n_feats_per_type;
	int num_total_feat_options = 0;

	{
		std::vector<RIFeatExtractor> dummy_feat_extractor(num_feat_types);
		n_feats_per_type.resize(num_feat_types);
		for(int ft = 0; ft < num_feat_types; ++ft)
		{
			dummy_feat_extractor[ft].initialise(cv::Size(10, 10), 8.0, J[ft], K[ft], M[ft], RIFeatExtractor::cmSpatial, false, RIFeatExtractor::comElementwise, feature_set_type[ft], max_rot_order, basis_type);
			n_feats_per_type[ft] = dummy_feat_extractor[ft].getNumDerivedFeats();
			if( (train_problem == ut::ptClassOri) || (train_problem == ut::ptClassOriPhase) )
			{
				dummy_feat_extractor[ft].getFeatsWithGivenR(1,ori_feats_lists[ft],true,Jmax[ft]);
				num_ori_feats_per_type[ft] = ori_feats_lists[ft].size();
			}
			if(Jmax[ft] >= 0)
			{
				dummy_feat_extractor[ft].getFeatsUsingLowJ(Jmax[ft],low_j_feats[ft]);
				num_total_feat_options += low_j_feats[ft].size();
			}
			else
				num_total_feat_options += n_feats_per_type[ft];
		}
	}

	const bool using_motion = any_of(feat_type.cbegin(), feat_type.cend(), [](ut::featType_t ft) { return (ft == ut::ftMotion); }) ;

	// Create a window for display
	if(display_mode)
	{
		omp_set_num_threads(1);
		namedWindow( "Display window", WINDOW_AUTOSIZE );
	}

	cout << "Precomputing features" << endl;

	// Read in each video and store the features needed
	#pragma omp parallel for firstprivate(previousvid)
	for(int v = 0; v < n_vids; ++v)
	{
		// Local variables within the parallel loop
		Mat_<unsigned char> I,I_prev;
		Mat_<float> processed_image[2];

		// Open the video file
		VideoCapture vid_obj;
		// Don't know why this is needed, but it crashes here without it...
		#pragma omp critical
		vid_obj.open( (viddir / uniquevidname[v]).string() );
		if (!vid_obj.isOpened())
		{
			cerr  << "Could not open reference " << (viddir / uniquevidname[v]).string() << endl;
			read_error = true;
			continue;
		}
		const int xsize = vid_obj.get(cv::CAP_PROP_FRAME_WIDTH);
		const int ysize = vid_obj.get(cv::CAP_PROP_FRAME_HEIGHT);
		float frame_rate = vid_obj.get(cv::CAP_PROP_FPS);

		if(isnan(frame_rate))
		{
			frame_rate = ut::getFrameRate(uniquevidname[v],viddir.string());
			if(isnan(frame_rate))
			{
				cerr << "Could not determine frame rate for video " << uniquevidname[v] << endl;
				read_error = true;
			}
		}

		// Resize the image so that the radius of the annotated label matches
		// the model's radius
		const float scale_factor = float(train_radius)/float(vidradius[v]);
		const int yresize = ysize*scale_factor;
		const int xresize = xsize*scale_factor;

		// Create objects for extracting feature representations for this video
		vector<imageFeatureProcessor> feat_processor(num_feat_types);
		for(int ft = 0; ft < num_feat_types; ++ft)
			feat_processor[ft].initialise(feat_type[ft],yresize,xresize,wl[ft],frame_rate);

		// Create a feature extraction object to be used with this video
		vector<RIFeatExtractor> feat_extractor(num_feat_types);
		for(int ft = 0; ft < num_feat_types; ++ft)
			feat_extractor[ft].initialise(cv::Size(xresize, yresize), 2*train_radius, J[ft], K[ft], M[ft], RIFeatExtractor::cmSpatial, use_memoiser[ft], RIFeatExtractor::comElementwise, feature_set_type[ft], max_rot_order, basis_type);

		vector<int> max_spat_basis_halfsize(num_feat_types);
		for(int ft = 0; ft < num_feat_types; ++ft)
			max_spat_basis_halfsize[ft] = feat_extractor[ft].getMaxSpatBasisHalfsize(Jmax[ft]);

		// Frame position counter (next to read in)
		int f = 0;

		// Loop through the frames in this video that appear in the
		// dataset
		for(int vd = 0 ; vd < int(datapoints_per_vid[v].size()); ++vd)
		{
			// Index of this data point in the full list
			const int d = datapoints_per_vid[v][vd];

			// Check whether this is the same frame as before
			if((v == previousvid) && (frameno[d] == frameno[d-1]))
			{
				// Do nothing - we already have the required images in the feature extractor
			}
			else
			{
				// Read in the relevant frame(s)
				// If using motion, we need to also have the frame before
				if( using_motion )
				{
					if(frameno[d] == 0)
						cerr << "No mechanism to cope with motion in frame 0, about to crash...!" << endl;
					// If reading two consecutive frames, we may have already read in the first frame we need
					if(f == frameno[d])
						I.copyTo(I_prev);
					else
					{
						Mat prev_image_in;
						// Set the video stream to read from the relevant point
						for( ; f < frameno[d]-1; ++f)
							vid_obj.grab();
						vid_obj >> prev_image_in; if(f != frameno[d] - 1) cout << "Warning: wrong frame number " << f << " " << frameno[d]-1 << endl;
						++f;
						cvtColor(prev_image_in,prev_image_in,cv::COLOR_BGR2GRAY);
						resize(prev_image_in,prev_image_in,Size(xresize,yresize));
						prev_image_in.convertTo(I_prev,CV_8U);
					}
				}

				Mat image_in;
				// Set the video stream to read from the relevant point
				for( ; f < frameno[d]; ++f)
					vid_obj.grab();
				vid_obj >> image_in; if(f != frameno[d]) cout << "Warning: wrong frame number " << f << " " << frameno[d] << endl;
				++f;
				resize(image_in,image_in,Size(xresize,yresize));
				cvtColor(image_in,image_in,cv::COLOR_BGR2GRAY);
				image_in.convertTo(I,CV_8U);

				// Transform the image to the desired representation
				// and place into the featureExtractor
				for(int ft = 0; ft < num_feat_types; ++ft)
				{
					feat_processor[ft].extract(I,processed_image[0],processed_image[1],&I_prev);
					if(feat_type[ft] == ut::ftInt)
						feat_extractor[ft].setScalarInputImage(processed_image[0]);
					else
						feat_extractor[ft].setVectorInputImage(processed_image[0],processed_image[1]);
				}

			} // end else (check for same frame)

			// Find the new centre of the annotation after the resize
			int centre_y_resize = centrey[d]*scale_factor;
			int centre_x_resize = centrex[d]*scale_factor;
			for(int ft = 0; ft < num_feat_types; ++ft)
			{
				if(centre_y_resize < max_spat_basis_halfsize[ft])
					centre_y_resize = max_spat_basis_halfsize[ft];
				else if(centre_y_resize >= yresize - max_spat_basis_halfsize[ft])
					centre_y_resize = yresize - max_spat_basis_halfsize[ft] - 1;
				if(centre_x_resize < max_spat_basis_halfsize[ft])
					centre_x_resize = max_spat_basis_halfsize[ft];
				else if(centre_x_resize >= xresize - max_spat_basis_halfsize[ft])
					centre_x_resize = xresize - max_spat_basis_halfsize[ft] - 1;
			}

			// Display the frames and labels
			if(display_mode)
			{
				Mat disp;
				Scalar colour;
				switch(label[d])
				{
					case 0:
						// Background
						colour = Scalar(255,255,255);
						break;
					case VIEW_4CHAM:
						colour = CLR_4CHAM;
						break;
					case VIEW_LVOT:
						colour = CLR_LVOT;
						break;
					case VIEW_RVOT:
						colour = CLR_RVOT;
						break;
					case VIEW_VSIGN:
						colour = CLR_VSIGN;
						break;
				}
				switch(feat_type[0])
				{
					case ut::ftInt :
						cvtColor(processed_image[0],disp,cv::COLOR_GRAY2BGR);
						disp /= 255.0;
						break;

					case ut::ftMotion :
						thesisUtilities::superimposeVector(I,feat_processor[0].getUnnormalisedFlow(),disp);
						break;

					default :
						normalize(processed_image[0],disp,0,1, cv::NORM_MINMAX);
						cvtColor(disp,disp,cv::COLOR_GRAY2BGR);
						break;
				} // end switch
				circle(disp,Point(centre_x_resize,centre_y_resize),train_radius,colour,1);
				imshow("Display window",disp);
				waitKey(0);
			}

			// Loop through and precalculate all the features for this example
			const cv::Point point(centre_x_resize, centre_y_resize);
			for(int ft = 0; ft < num_feat_types; ++ft)
			{
				featarray[ft][d].resize(n_feats_per_type[ft]);

				// Are we using only a subset of the features with low J values
				if(Jmax[ft] < 0)
				{
					for(int i = 0; i < n_feats_per_type[ft]; ++i )
					{
						feat_extractor[ft].getDerivedFeature(&point,(&point)+1,i,&(featarray[ft][d][i]));
						assert(!std::isnan(featarray[ft][d][i]));
					}
				}
				else
				{
					for(int i = 0; i < n_feats_per_type[ft]; ++i )
						featarray[ft][d][i] = nan("");
					for(int i : low_j_feats[ft])
					{
						feat_extractor[ft].getDerivedFeature(&point,(&point)+1,i,&(featarray[ft][d][i]));
						assert(!std::isnan(featarray[ft][d][i]));
					}
				}
			}

			// Sort out joint labels if required
			if(train_problem == ut::ptClassOri)
			{
				ori_labels[d].class_label = label[d];
				if(label[d] > 0)
					ori_labels[d].angle_label = orientation[d]*M_PI/180.0;
				else
					ori_labels[d].angle_label = 0.0;
			}
			else if(train_problem == ut::ptClassOriPhase)
			{
				ori_phase_labels[d].phase_label = cardiacphase[d];
				if(label[d] > 0)
					ori_phase_labels[d].angle_label = orientation[d]*M_PI/180.0;
				else
					ori_phase_labels[d].angle_label = 0.0;
			}

			// Calculate the orientation features if required
			if(train_problem == ut::ptClassOri || train_problem == ut::ptClassOriPhase)
			{
				if(label[d] > 0)
				{
					for(int ft = 0; ft < num_feat_types; ++ft)
					{
						sinorifeatsarray[ft][d].resize(num_ori_feats_per_type[ft]);
						cosorifeatsarray[ft][d].resize(num_ori_feats_per_type[ft]);
						for(int i = 0; i < num_ori_feats_per_type[ft]; ++i)
							feat_extractor[ft].getRawFeatureArg(&point,&(point)+1,ori_feats_lists[ft][i],cosorifeatsarray[ft][d].data()+i,sinorifeatsarray[ft][d].data()+i,true);
					}
				}
			}

			previousvid = v;

		} // end frame loop

		// Release the video object
		vid_obj.release();

	} // end video loop

	if(read_error)
	{
		cerr << "Finishing due to earlier read error in one parallel thread" << endl;
		return EXIT_FAILURE;
	}

	// Set up feature strings
	feature_header = "num_feat_types radius [feat_type J K M max_rot_order wl feature_set_type basis_type Jmax]";
	feat_stream << num_feat_types << " " << train_radius;
	for(int ft = 0; ft < num_feat_types; ++ft)
		feat_stream << " " << feat_type_string[ft] << " " << J[ft] << " " << K[ft] << " " << M[ft] << " " << max_rot_order << " " << wl[ft] << " " << feature_set_type[ft] << " " << basis_type << " " << Jmax[ft];
	feat_str = feat_stream.str();

	// Create a training functor object
	auto feature_lambda = [&] (auto first_id, const auto last_id, const std::array<int,2>& params, std::vector<float>::iterator out_it)
	{
		while(first_id != last_id)
		{
			const int id = *first_id;
			assert(!std::isnan(featarray[params[0]][id][params[1]]));
			*out_it++ = featarray[params[0]][id][params[1]];
			++first_id;
		}
	};


	// Lambda to generate parameter combinations for the forest
	std::default_random_engine rand_engine;
	std::random_device rd{};
	rand_engine.seed(rd());
	std::uniform_int_distribution<int> uni_dist;
	auto param_generator_lambda = [&] (std::array<int,2>& params)
	{
		// First decide which feature type to use
		params[0] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,num_feat_types-1});

		// Next decide which indivdiual feature to choose from those available for this feature type
		// This depends on whether we have to be careful to avoid features below a certain J value
		if(low_j_feats[params[0]].size() == 0 )
			params[1] = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,n_feats_per_type[params[0]]-1});
		else
		{
			const int choice_in_list = uni_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,int(low_j_feats[params[0]].size())-1});
			params[1] = low_j_feats[params[0]][choice_in_list];
		}
	};

	// Number of features to test at each node
	if(num_training_features < 0)
		num_training_features = num_total_feat_options/4;

	// Vector containing all the training ids
	// (Should be able to use a boost counting iterator here, but that crashes
	// for reasons I can't figure out)
	vector<int> train_ids(n_data);
	std::iota(train_ids.begin(),train_ids.end(),0);

	// Train and the classification forest and write to file
	if(train_problem == ut::ptClass || train_problem == ut::ptClassPhase || train_problem == ut::ptClassOriPhase)
	{
		cout << "Training Tree Classifier" << endl;
		cp::classifier<2> forest(n_class_labels,num_trees,tree_depth);
		forest.setClassNames(class_names);
		forest.train( train_ids.cbegin(), train_ids.cend(), label.cbegin(), feature_lambda, param_generator_lambda, num_training_features,true,0.5,fit_split_dists);
		forest.setFeatureDefinitionString(feature_header,feat_str);
		forest.writeToFile(modelfilename.string() + ".tr");
	}

	// Train further models depending on the problem type
	switch(train_problem)
	{
		case ut::ptClass :
			// Nothing more to be done
		break;

		case ut::ptClassOri :
		{
			// Create orientation functor object
			orientationTrainingFunctor ori_train_ftr(num_feat_types,sinorifeatsarray,cosorifeatsarray,ori_feats_lists);

			// Train the joint orientation forest and write to file
			cout << "Training Joint Forest Classifier/Orientation Regressor" << endl;
			cp::jointOrientationRegressor<cp::circCircSingleRegressor<orientationTrainingFunctor>,2> ori_forest(n_class_labels,&ori_train_ftr,num_trees,tree_depth);
			ori_forest.setClassNames(class_names);
			ori_forest.train( train_ids.cbegin(), train_ids.cend(), ori_labels.cbegin(), feature_lambda, param_generator_lambda, num_training_features,true,0.5,fit_split_dists);
			ori_forest.setFeatureDefinitionString(feature_header,feat_str);
			ori_forest.writeToFile(modelfilename.string() + "_ori.tr");
		}
		break;

		case ut::ptClassPhase :
			for(int c = 1; c < n_class_labels; ++c)
			{
				cout << "Training Phase Regressor Forest " << c << endl;
				cp::circularRegressor<2> phase_forest(num_trees, tree_depth);
				phase_forest.train(ids_per_label[c-1].cbegin(),ids_per_label[c-1].cend(), boost::make_permutation_iterator(cardiacphase.cbegin(),ids_per_label[c-1].cbegin()), feature_lambda, param_generator_lambda, num_training_features,true,0.5,fit_split_dists);
				phase_forest.setFeatureDefinitionString(feature_header,feat_str);
				phase_forest.writeToFile(modelfilename.string() + "_phase" + to_string(c) + ".tr");
			}
		break;

		case ut::ptClassOriPhase :
		{
			// Create orientation functor object
			orientationTrainingFunctor ori_train_ftr(num_feat_types,sinorifeatsarray,cosorifeatsarray,ori_feats_lists);

			for(int c = 1; c < n_class_labels; ++c)
			{
				cout << "Training Joint Phase/Orientation Regressor Forest " << c << endl;
				cp::jointOriPhaseRegressor<cp::circCircSingleRegressor<orientationTrainingFunctor>,2> ori_phase_forest(&ori_train_ftr,num_trees, tree_depth);
				const auto ori_phase_it =  boost::make_permutation_iterator(ori_phase_labels.cbegin(),ids_per_label[c-1].cbegin());
				ori_phase_forest.train(ids_per_label[c-1].cbegin(),ids_per_label[c-1].cend(), ori_phase_it, feature_lambda, param_generator_lambda, num_training_features,true,0.5,fit_split_dists);
				ori_phase_forest.setFeatureDefinitionString(feature_header,feat_str);
				ori_phase_forest.writeToFile(modelfilename.string() + "_phaseori" + to_string(c) + ".tr");
			}
		}
		break;

		default:
		break;

	} // end switch (train_problem)

	return EXIT_SUCCESS;
}
