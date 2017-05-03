#ifndef RFUTILITIES_H
#define RFUTILITIES_H

#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/core/core.hpp>

// Contains various functions and definitions that don't really fit anywhere else

namespace thesisUtilities
{
	// Data types
	enum featType_t : unsigned char
	{
		ftInt = 0,
		ftGrad,
		ftMotion,
		ftMGOF,
		ftInvalid
	};

	// Problem types
	enum problemType_t : unsigned char
	{
		ptClass = 0,
		ptClassPhase,
		ptClassOri,
		ptClassOriPhase,
		ptSubstructures,
		ptSubstructuresPCA,
		ptAbdomen
	};

	enum heartPresent_t : unsigned char
	{
		hpNone = 0,
		hpPresent,
		hpObscured
	};

	// Struct representing all the annotation information for one substructure
	struct subStructLabel_t
	{
		int x;
		int y;
		int ori;
		int present;
		bool labelled;
		subStructLabel_t() : x(-1), y(-1), ori(0), present(0), labelled(false) {}
	};

	// Prototypes
	bool readDataset(std::string filename, std::vector<std::string>& class_names, std::vector<std::string> &uniquevidname, std::vector< std::vector<int>> &datapoints_per_vid, std::vector<int> &frameno,
					std::vector<float> &vidradius, std::vector<int> &label, std::vector<float> &orientation_degrees, std::vector<int> &vidindex, std::vector<int> &centrey, std::vector<int> &centrex, std::vector<float> &cardiacphase);

	float getFrameRate(std::string filename,std::string viddir);

	featType_t strToFeat(std::string feat_type_string);

	problemType_t boolsToProblemType(const bool ori, const bool phase);

	bool parseFeatureDefinitionString(const std::string& feat_string, std::vector<int>& J, std::vector<int>& K, std::vector<int>& M, std::vector<int>& max_rot_order,
		                              std::vector<featType_t>& feat_type, std::vector<float>& wl, std::vector<int>& coupling_type, std::vector<int>& basis_type, std::vector<int>& Jmax, int& radius);

	bool parseSquareFeatureDefinitionString(const std::string& feat_string, int& ori_ind, int& n_orientations, int& num_feat_types, int& winhalfsize, int& featurehalfsize, std::vector<thesisUtilities::featType_t>& feat_type, std::vector<float>& wl, std::vector<int>& n_bins);

	bool readTrackFile(const std::string& filename, const int n_frames, bool& headup, int& radius, std::vector<bool>& labelled_track, std::vector<heartPresent_t>& heart_present_track,
		               std::vector<int>& centrey_track, std::vector<int>& centrex_track, std::vector<int>& ori_track_degrees, std::vector<int>& view_label_track, std::vector<int>& phase_point_track, std::vector<float>& cardiac_phase_track);

	bool readTrackFileRadiusOnly(const std::string& filename, int& radius);

	bool readAbdomenTrackFile(const std::string& filename, const int n_frames, bool& headup, int& radius, std::vector<bool>& labelled_track, std::vector<thesisUtilities::heartPresent_t>& abdomen_present_track,
		               std::vector<int>& centrey_track, std::vector<int>& centrex_track, std::vector<int>& ori_track);

	bool readSubstructuresTrackFile(const std::string& filename, const int n_frames, std::vector<std::string>& structure_names, std::vector<std::vector<subStructLabel_t>>& track);

	bool subStructureFileContains(const std::string& filename, const std::string& structname);

	bool trackFileContainsView(const std::string& filename, const int view);

	bool readGivenSubstructures(const std::string& filename, const std::vector<std::string>& structs, const int n_frames, std::vector<std::vector<subStructLabel_t>>& track);

	bool parseHoughPointFeatureDefinitionString(const std::string& feat_str, int& patchhalfsize, float& patch_radius_ratio);

	bool parseRIHoughFeatureDefinitionString(const std::string& featString, std::vector<int>& J, std::vector<int>& K, std::vector<int>& M, std::vector<int>& max_rot_order,
		                              std::vector<featType_t>& feat_type, std::vector<float>& wl, std::vector<int>& coupling_type, std::vector<int>& basis_type, std::vector<int>& Jmax, int& radius, float& patch_radius_ratio);

	bool checkFeaturesMatch(const std::string& feat_str, std::vector<int>& new_Jmax, const int train_radius, const std::vector<int>& J, const std::vector<int>& K,
							const std::vector<int>& M, const std::vector<int>& max_rot_order, const std::vector<int>& coupling_type,
							const std::vector<int>& basis_type, const std::vector<float>& wl, const std::vector<featType_t>& feat_type);

	bool checkHoughFeaturesMatch(const std::string& feat_str, std::vector<int>& new_Jmax, const int train_radius, const std::vector<int>& J, const std::vector<int>& K,
							     const std::vector<int>& M, const std::vector<int>& max_rot_order, const std::vector<int>& coupling_type,
							     const std::vector<int>& basis_type, const std::vector<float>& wl, const std::vector<featType_t>& feat_type);

	bool prepareMask(const std::string& filename, const cv::Size expected_size, cv::Mat_<unsigned char>& mask, const double shrink_distance = 0.0, const cv::Size output_size = cv::Size(-1,-1), const int boundary = -1 );

	void findValidPixels(std::vector<cv::Point>& pixels, const cv::Mat_<unsigned char>& mask, const int stride = 1, const int boundary = 0, std::vector<int>* const reverse = nullptr);

	template <typename TEigenMatrix>
	bool readEigenMatrix(std::ifstream& infile, TEigenMatrix& matrix)
	{
		for(int r = 0; r < matrix.rows(); ++r)
			for(int c = 0; c < matrix.cols(); ++c)
			{
				infile >> matrix(r,c);
				if(infile.fail())
				return false;
			}
		return true;
	}

	// Helper function that takes a vector of arguments and an expected number
	// It will fill an empty vector with a default value, or if there is one argument,
	// this will be copied to the others. Otherwise it returns false to show an error
	template <typename arg_type>
	bool pad_vector_args(std::vector<arg_type>& v, const int expected_number, const arg_type default_arg)
	{
		// Check to see whether it's OK as it is
		if(int(v.size()) == expected_number)
			return true;

		if(0 == v.size())
		{
			// Use the default value for all feature types
			for(int i = 0; i < expected_number; ++i)
				v.emplace_back(default_arg);
			return true;
		}
		else if (1 == v.size())
		{
			// Use the first element and copy to all the others
			for(int i = 1; i < expected_number; ++i)
				v.emplace_back(v.front());
			return true;
		}
		else
			return false;
	}

	// Wrap a floating point number to the range 0 to 2*pi
	template<typename T>
	T wrapTo2Pi(T x)
	{
		if(std::signbit(x))
			return 2.0*M_PI - std::fmod(-x,2.0*M_PI);
		else
			return std::fmod(x,2.0*M_PI);
	}

	// Wrap each element in an opencv image to the range 0 to 2*pi
	template<typename T>
	void wrapMatTo2Pi(cv::Mat_<T>& in)
	{
		for( int i = 0; i < in.rows; ++i)
		{
			T* p = (T*)in.ptr(i);
			for ( int j = 0; j < in.cols; ++j)
			{
				p[j] = wrapTo2Pi(p[j]);
			}
		}
	}

}

// inclusion guard
#endif
