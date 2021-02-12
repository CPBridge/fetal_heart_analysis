#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <omp.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "squareTrainingFunctors.h"
#include <canopy/classifier/classifier.hpp>
#include <canopy/circularRegressor/circularRegressor.hpp>
#include "thesisUtilities.h"
#include "histograms.h"
#include "imageFeatureProcessor.h"
#include "displayFunctions.h"


int main( int argc, char** argv )
{
	// Namespaces
	using namespace cv;
	using namespace std;
	namespace ut = thesisUtilities;
	namespace po = boost::program_options;
	namespace fs = boost::filesystem;
	namespace cp = canopy;

	// Default parameters
	constexpr int C_DEFAULT_TREES = 8;
	constexpr int C_DEFAULT_LEVELS = 10;
	const string C_DEFAULT_FEAT = "int";
	constexpr float C_DEFAULT_WAVELENGTH = 0.0;
	constexpr int C_DEFAULT_BINS = 4;
	constexpr int C_DEFAULT_ORIENTATIONS = 8;
	constexpr int C_DEFAULT_NUM_TRAINING_FEATURES = 3000;

	// Declarations
	Mat I_prev, I_prev_rotated;
	vector<string> uniquevidname;
	vector<string> class_names;
	vector<int> vidindex,centrey,centrex;
	vector<int> frameno;
	vector<float> orientation;
	vector<int> label;
	vector<float> vidradius;
	vector<float> cardiacphase;
	vector< vector<int> > datapoints_per_vid;
	bool read_error = false;
	unsigned previousvid = -1;
	int n_class_labels,winhalfsize,featurehalfsize,n_orientations;
	fs::path datasetfile, viddir, modelfilename;

	// Forest parameters
	int num_trees,tree_depth,num_training_features;

	// Description Parameters
	vector<string> feat_type_string;
	vector<float> wl;
	vector<int> n_bins;

	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("dataset,d", po::value<fs::path>(&datasetfile), "input dataset file")
		("videodirectory,v", po::value<fs::path>(&viddir)->default_value("./"), "directory containing video files")
		("trees,n",po::value<int>(&num_trees)->default_value(C_DEFAULT_TREES), "number of trees")
		("depth,l",po::value<int>(&tree_depth)->default_value(C_DEFAULT_LEVELS), "number of levels per tree")
		("num_training_features,t",po::value<int>(&num_training_features)->default_value(C_DEFAULT_NUM_TRAINING_FEATURES), "number of features to consider at each split node")
		("phase,p", "also train a phase regression forest")
		("phase_only,P", "only train the phase forest, not the detection/classification forests")
		("orientations,O",po::value<int>(&n_orientations)->default_value(C_DEFAULT_ORIENTATIONS), "number of orientations at which to train detectors")
		("winhalfsize,r",po::value<int>(&winhalfsize)->default_value(30), "window half size at which to detect structures in the trained model")
		("featurehalfsize,x",po::value<int>(&featurehalfsize)->default_value(-1), "the actual features will be taken within a window of this half size, must be <= winhalfsize, default is equal to winhalfsize")
		("featureType,f",po::value<vector<string>>(&feat_type_string)->multitoken(), "type of feature used (multiple arguments may be listed")
		("bins,b",po::value<vector<int>>(&n_bins)->multitoken(), "number of orientation histogram bins")
		("wavelength,w",po::value<vector<float>>(&wl)->multitoken(), "monogenic filter centre-wavelength")
		("without_split_node_dists,s" , "do not fit node distributions to split nodes in the forest")
		("angle_jitter,j", "add random noise to the angle labels so they cover the whole orientation bin")
		("modelfile,o", po::value<fs::path>(&modelfilename)->default_value("modelout"), "root name for output files")
		("display,D", "input dataset file");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		cout << "Train a random forest using rectangle filters" << endl;
		cout << desc << endl;
		return EXIT_SUCCESS;
	}

	if (vm.count("dataset") != 1)
	{
		cout << "No dataset file specified (use -d option)" << endl;
		return EXIT_FAILURE;
	}

	const bool train_phase = bool(vm.count("phase"));
	const bool phase_only = bool(vm.count("phase_only"));
	const bool fit_split_dists = !bool(vm.count("without_split_node_dists"));
	const bool display = bool(vm.count("display"));
	const bool angle_jitter = bool(vm.count("angle_jitter"));

	if(!train_phase && phase_only)
	{
		cerr << "Incompatible options, must be training a phase model to use the --phase_only (-P) option, otherwise there's nothing to train!" << endl;
		return EXIT_FAILURE;
	}

	// Interpret the feature type strings
	const int num_feat_types = feat_type_string.size();
	if(num_feat_types == 0)
	{
		cerr << "No feature type specified" << endl;
		return EXIT_FAILURE;
	}
	vector<ut::featType_t> feat_type;
	feat_type.resize(num_feat_types);
	transform(feat_type_string.begin(),feat_type_string.end(),feat_type.begin(),ut::strToFeat);

	for(auto ft : feat_type)
	{
		if(ut::ftInvalid == ft)
		{
			cerr << "ERROR: Invalid feature type argument" << endl;
			return EXIT_FAILURE;
		}
	}

	if(n_orientations < 1)
	{
		cerr << "ERROR: Invalid number of orientations " << n_orientations << endl;
		return EXIT_FAILURE;
	}
	const float orientation_spacing = 2.0*M_PI/n_orientations;
	vector<float> detector_orientation_list(n_orientations);
	for(int ori_ind = 0; ori_ind < n_orientations; ++ori_ind)
		detector_orientation_list[ori_ind] = ori_ind*orientation_spacing;

	if(!ut::pad_vector_args<float>(wl,num_feat_types,C_DEFAULT_WAVELENGTH))
	{
		cerr << "ERROR: Unexpected number of wavelength parameters" << endl;
		return EXIT_FAILURE;
	}

	if(winhalfsize <= 0)
	{
		cerr << "ERROR: Invalid winhalfsize parameter, you said " << winhalfsize << endl;
		return EXIT_FAILURE;
	}
	if(featurehalfsize < 0)
	{
		featurehalfsize = winhalfsize;
	}
	else if(featurehalfsize > winhalfsize)
	{
		cerr << "ERROR: The value for featurehalfsize must be less than or equal to winhalfsize" << endl;
		return EXIT_FAILURE;
	}


	// Split the feature types into scalar and vector
	vector<ut::featType_t> scalar_feat_types;
	vector<ut::featType_t> vector_feat_types;
	vector<float> scalar_wavelengths;
	vector<float> vector_wavelengths;
	for( int ft = 0; ft < num_feat_types; ++ft)
	{
		if (wl[ft] < 0.0)
		{
			cerr << "Must use a positive wavelength" << endl;
			return EXIT_FAILURE;
		}
		if(feat_type[ft] == ut::ftInt)
		{
			scalar_feat_types.emplace_back(feat_type[ft]);
			scalar_wavelengths.emplace_back(wl[ft]);
		}
		else
		{
			vector_feat_types.emplace_back(feat_type[ft]);
			vector_wavelengths.emplace_back(wl[ft]);
		}
	}
	const int num_scalar_feat_types = scalar_feat_types.size();
	const int num_vector_feat_types = vector_feat_types.size();

	if(!ut::pad_vector_args<int>(n_bins,num_vector_feat_types,C_DEFAULT_BINS))
	{
		cerr << "Unexpected number of bin parameters" << endl;
		return EXIT_FAILURE;
	}

	// Check for sensible input values
	for(int ft = 0; ft < num_vector_feat_types; ++ft)
	{
		if(n_bins[ft] <= 0)
		{
			cerr << "Invalid number of bins " << n_bins[ft] << endl;
			return EXIT_FAILURE;
		}
	}

	const bool using_motion = any_of(feat_type.begin(), feat_type.end(), [](ut::featType_t ft) { return (ft == ut::ftMotion); }) ;

	// Read the dataset file into arrays
	if(!ut::readDataset(datasetfile.string(),class_names,uniquevidname,datapoints_per_vid,frameno,vidradius,label,orientation,vidindex,centrey,centrex,cardiacphase))
	{
		cout << "Error reading file '" << datasetfile << "'." << endl;
		return EXIT_FAILURE;
	}
	const int n_data = vidindex.size();
	n_class_labels = 1;
	for(int d = 0; d < n_data; ++d)
		if(label[d] > n_class_labels - 1)
			n_class_labels = label[d] + 1;

	// Prepare a random number generator
	std::default_random_engine rand_engine;
	std::uniform_real_distribution<float> uni_real_dist(-orientation_spacing/2.0,orientation_spacing/2.0);

	// Convert the orientations to radians to use from here on
	for(float& o : orientation)
	{
		o *= (M_PI/180.0);

		// Add angle jitter
		if(angle_jitter)
		{
			o += uni_real_dist(rand_engine);
			o = thesisUtilities::wrapTo2Pi(o);
		}
	}

	// Sort out the data points into vectors of different view classes
	vector<vector<int>> ids_per_label;
	if( train_phase )
	{
		ids_per_label.resize(n_class_labels-1);
		for(int d = 0; d < n_data; ++d)
			if(label[d] > 0)
				ids_per_label[label[d]-1].emplace_back(d);
	}

	uni_real_dist = std::uniform_real_distribution<float>(0.0,2.0*M_PI);

	// Create a window for display
	if (display)
		namedWindow( "Display window", WINDOW_AUTOSIZE );

	// Loop over training orientations
	// This arrangement means looping through the videos for every training
	// orientation, however it dramatically reduces memory usage allowing larger
	// forests to be trained or bigger datasets to be used
	for(int ori_ind = 0; ori_ind < n_orientations; ++ori_ind)
	{

		// Create array of mats for scalar representations
		vector<Mat_<int>> scalar_images_array;
		if(num_scalar_feat_types > 0)
			scalar_images_array.resize(n_data);

		// Create arrays of Mats for storing the vector images
		vector<vector<vector<Mat_<float>>>> vector_images_ary;
		if(num_vector_feat_types > 0)
		{
			vector_images_ary.resize(n_data);
			for(auto& v_data : vector_images_ary)
			{
				v_data.resize(num_vector_feat_types);
				for(int ft = 0; ft < num_vector_feat_types; ++ft)
					v_data[ft].resize(n_bins[ft]);
			}
		}

		// Progress message
		cout << "Loading in training samples for orientation " << ori_ind << endl;

		// Read in each video and store the frames needed
		#pragma omp parallel for firstprivate(previousvid)
		for(unsigned v = 0; v < uniquevidname.size(); ++v)
		{
			Mat_<unsigned char> I,I_prev;

			// Open the video file
			VideoCapture vid_obj( (viddir / uniquevidname[v]).string() );
			if (!vid_obj.isOpened())
			{
				cout  << "Could not open reference " << viddir / uniquevidname[v] << endl;
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
					cout << "Could not determine frame rate for video " << uniquevidname[v] << endl;
					read_error = true;
				}
			}

			const float scale_factor = float(winhalfsize)/float(vidradius[v]);
			const int yresize = ysize*scale_factor;
			const int xresize = xsize*scale_factor;

			// Create objects for extracting feature representations for this video
			vector<imageFeatureProcessor> feat_processor_vector(num_feat_types);
			for(int ft = 0; ft < num_vector_feat_types; ++ft)
			{
				if(vector_feat_types[ft] != ut::ftInt)
					feat_processor_vector[ft].initialise(vector_feat_types[ft],yresize,xresize,vector_wavelengths[ft],frame_rate);
			}

			// Frame position counter
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

				} // end else (check for same frame)

				float ori_offset;

				// Choose a random orientation for background classes
				if(label[d] == 0)
					ori_offset = uni_real_dist(rand_engine);
				// For other classes correct for the orientation of the heart
				else
					ori_offset = -orientation[d] + detector_orientation_list[ori_ind];

				// Rotation matrix to correct for orientation of the image
				Mat rot_mat = getRotationMatrix2D(Point2f(centrex[d]*scale_factor,centrey[d]*scale_factor),ori_offset*(180.0/M_PI),1.0);

				// Rectangle defining the window of interest in the image
				int winleft = std::round(centrex[d]*scale_factor-featurehalfsize);
				int wintop = std::round(centrey[d]*scale_factor-featurehalfsize);
				if(winleft < 0)
					winleft = 0;
				else if(winleft + 2*featurehalfsize+1 >= xresize)
					winleft = xresize - 2*featurehalfsize - 2;
				if(wintop < 0)
					wintop = 0;
				else if(wintop + 2*featurehalfsize+1 >= yresize)
					wintop = yresize - 2*featurehalfsize - 2;
				cv::Rect window_rect = Rect(winleft,wintop,2*featurehalfsize+1,2*featurehalfsize+1);

				// Loop over all scalar feature representations
				for(int ft = 0; ft < num_scalar_feat_types; ++ft)
				{
					// Rotate the image
					Mat_<unsigned char> I_rotated;
					cv::warpAffine(I,I_rotated,rot_mat,I.size());
					Mat_<unsigned char> win = I_rotated(window_rect).clone();

					// Display
					if(display)
					{
						cv::imshow( "Display window", win );
						cout << "Sample " << d << ", class " << label[d] << ", orientation " << detector_orientation_list[ori_ind] << endl;
						cv::waitKey(0);
					}

					// Place the integral image of the patch into the array
					cv::integral(win,scalar_images_array[d]);
				}

				// Loop over all vector feature representations
				for(int ft = 0; ft < num_vector_feat_types; ++ft)
				{
					Mat_<float> mag, mag_rotated, ori, ori_rotated;

					// Transform to the relevant representation
					feat_processor_vector[ft].extract(I,mag,ori,&I_prev);

					// Rotate the magnitiude image
					cv::warpAffine(mag,mag_rotated,rot_mat,mag.size());

					// Rotate the orientation image
					cv::warpAffine(ori,ori_rotated,rot_mat,ori.size());
					ori_rotated += ori_offset;
					thesisUtilities::wrapMatTo2Pi(ori_rotated);

					Mat_<float> mag_win = mag_rotated(window_rect).clone();
					Mat_<float> ori_win = ori_rotated(window_rect).clone();

					// Display
					if(display)
					{
						Mat planes[2];
						Mat vector_im;
						cv::polarToCart(mag_win,ori_win,planes[0],planes[1]);
						cv::merge(planes,2,vector_im);
						Mat disp;
						Mat_<Vec3b> blank = Mat_<Vec3b>::zeros(mag_win.size());
						thesisUtilities::superimposeVector(blank,vector_im,disp,0.001,2);
						cout << "Sample " << d << ", class " << label[d] << ", orientation " << detector_orientation_list[ori_ind] << endl;
						cv::imshow( "Display window", disp );
						cv::waitKey(0);
					}

					// Find the orientation histogram
					softorihist(ori_win,mag_win,n_bins[ft],vector_images_ary[d][ft].data());
				}

				previousvid = v;
			}

			// Release the video object
			vid_obj.release();
		}

		if(read_error)
		{
			cout << "Finishing due to earlier read error in one parallel thread" << endl;
			return EXIT_FAILURE;
		}

		const string feature_header = "ori_ind n_orientations num_feat_types winhalfsize featurehalfsize [feat_type wl n_bins]*num_feat_types";

		// Create a functor object
		squareTrainingFunctorMixed train_ftr(scalar_images_array,vector_images_ary,featurehalfsize);

		const auto param_generator_lamda = [&] (std::array<int,rectangleFilterDefines::NUM_RECT_PARAMS>& params) {train_ftr.generateParameters(params);};

		// Strings to hold feature information
		stringstream feat_stream;
		feat_stream << ori_ind << " " << n_orientations << " " << num_feat_types << " " << winhalfsize << " " << featurehalfsize << " ";
		for(int ft = 0; ft < num_scalar_feat_types; ++ft)
			feat_stream << feat_type_string[ft] << " " << wl[ft] << " " << "0 ";
		for(int ft = 0; ft < num_vector_feat_types; ++ft)
			feat_stream << feat_type_string[ft-num_scalar_feat_types] << " " << wl[ft-num_scalar_feat_types] << " " << n_bins[ft] << " ";
		feat_stream.str() = feat_stream.str();

		if(!phase_only)
		{
			// Create the random forest object
			cp::classifier<rectangleFilterDefines::NUM_RECT_PARAMS> forest(n_class_labels,num_trees,tree_depth);

			// Vector containing all the training ids
			// (Should be able to use a boost counting iterator here, but that crashes
			// for reasons I can't figure out)
			vector<int> train_ids(n_data);
			std::iota(train_ids.begin(),train_ids.end(),0);

			// Train and write to file
			cout << "Training classification forest..." << endl;
			forest.setClassNames(class_names);
			forest.train(train_ids.cbegin(), train_ids.cend(), label.cbegin(), train_ftr, param_generator_lamda, num_training_features,true,0.5,fit_split_dists);
			forest.setFeatureDefinitionString(feature_header,feat_stream.str());
			forest.writeToFile(modelfilename.string() + "_rotation_" + std::to_string(ori_ind) + ".tr");
		}

		// Train phase regression forests
		if(train_phase)
		{
			for(int c = 1; c < n_class_labels; ++c)
			{
				cout << "Training Phase Regressor Forest " << c << endl;
				cp::circularRegressor<rectangleFilterDefines::NUM_RECT_PARAMS> phase_forest(num_trees, tree_depth);
				phase_forest.train(ids_per_label[c-1].cbegin(),ids_per_label[c-1].cend(), boost::make_permutation_iterator(cardiacphase.cbegin(),ids_per_label[c-1].cbegin()), train_ftr, param_generator_lamda, num_training_features,true,0.5,fit_split_dists);
				phase_forest.setFeatureDefinitionString(feature_header,feat_stream.str());
				phase_forest.writeToFile(modelfilename.string() + "_rotation_" + std::to_string(ori_ind) + "_phase_" + to_string(c) + ".tr");
			}
		}
	}

	return EXIT_SUCCESS;
}
