#include "thesisUtilities.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#define FRAME_RATE_DATABASE "frameratedatabase"

using namespace std;

namespace thesisUtilities
{


// Function to read in a dataset file, returns true if successful
bool readDataset(string filename,vector<string>& class_names,vector<string> &uniquevidname, vector< vector<int> > &datapoints_per_vid, vector<int> &frameno,
				vector<float> &vidradius, vector<int> &label, vector<float> &orientation_degrees, vector<int> &vidindex, vector<int> &centrey, vector<int> &centrex, vector<float> &cardiacphase)
{
	string dummy_string;
	int dummy_int;
	float dummy_float;
	vector<int> vidcount;
	int n_data,d;
	unsigned int v;
	bool unique;

	// Clear any existing contents
	uniquevidname.clear();
	datapoints_per_vid.clear();
	frameno.clear();
	vidradius.clear();
	label.clear();
	orientation_degrees.clear();
	vidindex.clear();
	centrex.clear();
	centrey.clear();
	cardiacphase.clear();

	// Open the text file and check it opened ok
	ifstream infile(filename.c_str());
	if (!infile.is_open())
		return false;

	// Skip the first line - it's a header line
	getline(infile,dummy_string);

	// Next line is the list of class names
	getline(infile,dummy_string);
	stringstream ss(dummy_string);
	while(ss)
	{
		string temp;
		ss >> temp;
		class_names.emplace_back(temp);
	}

	// The first number should be the number of lines in the file
	infile >> n_data;
	if(infile.fail())
		return false;

	// Create the output data arrays
	vidindex.reserve(n_data);
	frameno.reserve(n_data);
	centrey.reserve(n_data);
	centrex.reserve(n_data);
	label.reserve(n_data);

	// Read in lines in a loop
	for(d = 0; d < n_data; ++d)
	{
		// First comes the video filename
		infile >> dummy_string;
		if(infile.fail())
			return false;

		// See if this is a unique video name
		unique = true;
		for(v = 0 ; v < uniquevidname.size() ; ++v)
		{
			if(dummy_string.compare(uniquevidname[v]) == 0)
				// The two strings are equal (we've seen this vid before)
			{
				unique = false;
				vidindex.emplace_back(v);
				vidcount[v]++;
				datapoints_per_vid[v].emplace_back(d);
				break;
			}
		}

		// If it is unique, add this to the list
		if(unique)
		{
			uniquevidname.emplace_back(dummy_string);
			vidindex.emplace_back(uniquevidname.size()-1);
			vidcount.emplace_back(1);
			vidradius.emplace_back(0.0);
			datapoints_per_vid.emplace_back(vector<int>(1,d));
		}

		// Next comes the frame number
		infile >> dummy_int;
		if(infile.fail())
			return false;
		frameno.emplace_back(dummy_int);

		// Radius
		infile >> dummy_float;
		if(infile.fail())
			return false;
		vidradius[vidindex[d]] += dummy_float;

		// Window centre
		infile >> dummy_int;
		if(infile.fail())
			return false;
		centrey.emplace_back(dummy_int);
		infile >> dummy_int;
		if(infile.fail())
			return false;
		centrex.emplace_back(dummy_int);

		// Label
		infile >> dummy_int;
		if(infile.fail())
			return false;
		label.emplace_back(dummy_int);

		// Orientation
		infile >> dummy_int;
		if(infile.fail())
			return false;
		orientation_degrees.emplace_back(float(dummy_int));

		// Cardiac phase
		infile >> dummy_float;
		if(infile.fail())
			return false;
		cardiacphase.emplace_back(dummy_float);

	}

	// Take average of all the radii for each video to find the video radius
	for(v = 0 ; v < uniquevidname.size() ; ++v)
	{
		vidradius[v] /= float(vidcount[v]);
	}

	// Close file and return successfully
	infile.close();
	return true;
}


// Utility function to read in the frame rate from a database file
// Used because sometimes opencv cannot find the correct frame rate
float getFrameRate(string filename, string viddir)
{
	string vidname;
	float frame_rate = nan(""), temp;

	filename = filename.substr(filename.find_last_of("/")+1,string::npos);

	// Append trailing slash if necessary
	if(viddir.at(viddir.length() -1 ) != '/')
		viddir += '/';

	ifstream infile(viddir + FRAME_RATE_DATABASE);
	if(!infile.is_open())
	{
		cout << "Could not open frame rate database file " << viddir + FRAME_RATE_DATABASE << endl;
		return frame_rate;
	}

	while (infile >> vidname)
	{
		infile >> temp;

		if(filename.compare(vidname) == 0)
		{
			frame_rate = temp;
			break;
		}
	}

	infile.close();

	return frame_rate;

}

// Takes a string and converts it to a featType_t describing which features to use
featType_t strToFeat(std::string feat_type_string)
{
	if(feat_type_string.compare("int") == 0)
		return ftInt;
	else if(feat_type_string.compare("mgof") == 0)
		return ftMGOF;
	else if(feat_type_string.compare("grad") == 0)
		return ftGrad;
	else if(feat_type_string.compare("motion") == 0)
		return ftMotion;

	return ftInvalid;
}

problemType_t boolsToProblemType(const bool ori, const bool phase)
{
	if(ori && !phase)
		return ptClassOri;
	else if(!ori && phase)
		return ptClassPhase;
	else if(ori && phase)
		return ptClassOriPhase;
	else
		return ptClass;
}

bool parseFeatureDefinitionString(const string& feat_string, vector<int>& J, vector<int>& K, vector<int>& M, vector<int>& max_rot_order,
	                              vector<featType_t>& feat_type, vector<float>& wl, vector<int>& coupling_type, vector<int>& basis_type, vector<int>& Jmax, int& radius)
{
	// Clear all the vectors
	J.clear();	K.clear();	M.clear(); max_rot_order.clear(); Jmax.clear();
	feat_type.clear(); wl.clear(); coupling_type.clear(); basis_type.clear();

	// A string stream for parsing the string
	stringstream ss;
	ss.str(feat_string);

	int num_feat_types;
	ss >> num_feat_types;
	if(ss.fail())
		return false;

	ss >> radius;
	if(ss.fail())
		return false;

	for(int ft = 0; ft < num_feat_types; ++ft)
	{
		string tempstring;
		int tempint;
		float tempfloat;

		// Feature type
		ss >> tempstring;
		if(ss.fail())
			return false;
		feat_type.emplace_back(strToFeat(tempstring));

		// J
		ss >> tempint;
		if(ss.fail())
			return false;
		J.emplace_back(tempint);

		// K
		ss >> tempint;
		if(ss.fail())
			return false;
		K.emplace_back(tempint);

		// M
		ss >> tempint;
		if(ss.fail())
			return false;
		M.emplace_back(tempint);

		// max_rot_order
		ss >> tempint;
		if(ss.fail())
			return false;
		max_rot_order.emplace_back(tempint);

		// wl
		ss >> tempfloat;
		if(ss.fail())
			return false;
		wl.emplace_back(tempfloat);

		// coupling_type
		ss >> tempint;
		if(ss.fail())
			return false;
		coupling_type.emplace_back(tempint);

		// basis_type
		ss >> tempint;
		if(ss.fail())
			return false;
		basis_type.emplace_back(tempint);

		// Jmax
		ss >> tempint;
		if(ss.fail())
			return false;
		Jmax.emplace_back(tempint);

	}

	return true;

}

bool parseRIHoughFeatureDefinitionString(const string& featString, vector<int>& J, vector<int>& K, vector<int>& M, vector<int>& max_rot_order,
	                              vector<featType_t>& feat_type, vector<float>& wl, vector<int>& coupling_type, vector<int>& basis_type, vector<int>& Jmax, int& radius, float& patch_radius_ratio)
{
	// Clear all the vectors
	J.clear();	K.clear();	M.clear(); max_rot_order.clear();

	// A string stream for parsing the string
	stringstream ss;
	ss.str(featString);

	int num_feat_types;
	ss >> num_feat_types;
	if(ss.fail())
		return false;

	ss >> radius;
	if(ss.fail())
		return false;

	ss >> patch_radius_ratio;
	if(ss.fail())
		return false;

	for(int ft = 0; ft < num_feat_types; ++ft)
	{
		string tempstring;
		int tempint;
		float tempfloat;

		// Feature type
		ss >> tempstring;
		if(ss.fail())
			return false;
		feat_type.emplace_back(strToFeat(tempstring));

		// J
		ss >> tempint;
		if(ss.fail())
			return false;
		J.emplace_back(tempint);

		// K
		ss >> tempint;
		if(ss.fail())
			return false;
		K.emplace_back(tempint);

		// M
		ss >> tempint;
		if(ss.fail())
			return false;
		M.emplace_back(tempint);

		// max_rot_order
		ss >> tempint;
		if(ss.fail())
			return false;
		max_rot_order.emplace_back(tempint);

		// wl
		ss >> tempfloat;
		if(ss.fail())
			return false;
		wl.emplace_back(tempfloat);

		// coupling_type
		ss >> tempint;
		if(ss.fail())
			return false;
		coupling_type.emplace_back(tempint);

		// basis_type
		ss >> tempint;
		if(ss.fail())
			return false;
		basis_type.emplace_back(tempint);

		// Jmax
		ss >> tempint;
		if(ss.fail())
			return false;
		Jmax.emplace_back(tempint);

	}

	return true;

}

bool parseSquareFeatureDefinitionString(const std::string& feat_string, int& ori_ind, int& n_orientations, int& num_feat_types, int& winhalfsize, int& featurehalfsize, std::vector<thesisUtilities::featType_t>& feat_type, std::vector<float>& wl, std::vector<int>& n_bins)
{
	std::stringstream ss;
	ss.str(feat_string);

	// Clear vectors
	feat_type.clear();
	wl.clear();
	n_bins.clear();

	ss >> ori_ind;
	if(ss.fail())
		return false;

	ss >> n_orientations;
	if(ss.fail())
		return false;

	ss >> num_feat_types;
	if(ss.fail())
		return false;

	ss >> winhalfsize;
	if(ss.fail())
	return false;

	ss >> featurehalfsize;
	if(ss.fail())
	return false;

	for(int f = 0; f < num_feat_types; ++f)
	{
		std::string dummy_string;
		float dummy_float;
		int dummy_int;

		// Read the parameters in one by one
		ss >> dummy_string;
		if(ss.fail())
			return false;
		feat_type.emplace_back(thesisUtilities::strToFeat(dummy_string));

		ss >> dummy_float;
		if(ss.fail())
			return false;
		wl.emplace_back(dummy_float);

		ss >> dummy_int;
		if(ss.fail())
			return false;
		n_bins.emplace_back(dummy_int);
	}

	return true;

}


// Wrap an integer angle in degrees to the range 0 to 359
void wrapTo360(int& a)
{
	while(a < 0) a += 360;
	while(a >= 360) a -= 360;
}



bool readTrackFile(const string& filename, const int n_frames, bool& headup, int& radius, vector<bool>& labelled_track, vector<heartPresent_t>& heart_present_track,
	               vector<int>&  centrey_track, vector<int>&  centrex_track, vector<int>&  ori_track_degrees, vector<int>&  view_label_track, vector<int>&  phase_point_track, vector<float>& cardiac_phase_track)
{
	// Resize the arrays
	labelled_track.resize(n_frames);
	heart_present_track.resize(n_frames);
	centrex_track.resize(n_frames);
	centrey_track.resize(n_frames);
	ori_track_degrees.resize(n_frames);
	view_label_track.resize(n_frames);
	phase_point_track.resize(n_frames);
	cardiac_phase_track.resize(n_frames);

	ifstream infile(filename.c_str());
	if (infile.is_open())
	{
		string dummy_string;

		// Skip the first and second lines - a header line
		// and video dimensions respectively
		getline(infile,dummy_string);
		getline(infile,dummy_string);

		infile >> headup;
		if(infile.fail())
			return false;

		infile >> radius;
		if(infile.fail())
			return false;

		for(int f = 0; f < n_frames; f++)
		{
			int dummy_int;
			infile >> dummy_int;
			if(infile.fail() || dummy_int != f)
			{
				if(infile.eof()) // we've reached the end of the file before we expected to, mark the other frames as unlabelled
				{
					for(int l = f; l < n_frames; ++l)
					{
						labelled_track[l] = false;
						heart_present_track[f] = hpNone;
						centrey_track[l] = 0;
						centrex_track[l] = 0;
						ori_track_degrees[l] = 0;
						view_label_track[l] = 0;
						phase_point_track[l] = 0;
						cardiac_phase_track[l] = 0;
					}
					break;
				}
				else
					return false;
			}

			bool tempbool;
			infile >> tempbool;
			if(infile.fail())
				return false;
			labelled_track[f] = tempbool;

			infile >> dummy_int;
			if(infile.fail())
				return false;
			heart_present_track[f] = heartPresent_t(dummy_int);

			infile >> centrey_track[f];
			if(infile.fail())
				return false;

			infile >> centrex_track[f];
			if(infile.fail())
				return false;

			infile >> ori_track_degrees[f];
			if(infile.fail())
				return false;
			// Wrap to 360 degrees
			wrapTo360(ori_track_degrees[f]);

			infile >> view_label_track[f];
			if(infile.fail())
				return false;

			infile >> phase_point_track[f];
			if(infile.fail())
				return false;

			infile >> cardiac_phase_track[f];
			if(infile.fail())
				return false;

		}

		infile.close();

		return true;
	}
	else
		return false;
}

// Read in just the radius from a heart trackfile
bool readTrackFileRadiusOnly(const std::string& filename, int& radius)
{
	ifstream infile(filename.c_str());
	if (infile.is_open())
	{
		string dummy_string;

		// Skip the first and second lines - a header line
		// and video dimensions respectively
		getline(infile,dummy_string);
		getline(infile,dummy_string);

		bool headup;
		infile >> headup;
		if(infile.fail())
			return false;

		infile >> radius;
		if(infile.fail())
			return false;

		return true;
	}
	else
		return false;
}



// Read in a track file of abdomen data to arrays
bool readAbdomenTrackFile(const string& filename, const int n_frames, bool& headup, int& radius, vector<bool>& labelled_track, vector<thesisUtilities::heartPresent_t>& abdomen_present_track,
	               vector<int>& centrey_track, vector<int>& centrex_track, vector<int>& ori_track)
{

	ifstream infile(filename.c_str());
	if (infile.is_open())
	{
		labelled_track.resize(n_frames);
		centrey_track.resize(n_frames);
		centrex_track.resize(n_frames);
		abdomen_present_track.resize(n_frames);
		ori_track.resize(n_frames);


		string dummy_string;

		// Skip the first and second lines - a header line
		// and video dimensions respectively
		getline(infile,dummy_string);
		getline(infile,dummy_string);

		infile >> headup;
		if(infile.fail())
			return false;

		infile >> radius;
		if(infile.fail())
			return false;

		for(int f = 0; f < n_frames; f++)
		{
			int dummy_int;
			infile >> dummy_int;
			if(infile.fail() || dummy_int != f)
			{
				if(infile.eof()) // we've reached the end of the file before we expected to, mark the other frames as unlabelled
				{
					for(int l = f; l < n_frames; ++l)
					{
						labelled_track[l] = false;
						abdomen_present_track[f] = hpNone;
						centrey_track[l] = 0;
						centrex_track[l] = 0;
						ori_track[l] = 0;
					}
					break;
				}
				else
					return false;
			}

			bool tempbool;
			infile >> tempbool;
			if(infile.fail())
				return false;
			labelled_track[f] = tempbool;

			infile >> dummy_int;
			if(infile.fail())
				return false;
			abdomen_present_track[f] = thesisUtilities::heartPresent_t(dummy_int);

			infile >> centrey_track[f];
			if(infile.fail())
				return false;

			infile >> centrex_track[f];
			if(infile.fail())
				return false;

			infile >> ori_track[f];
			if(infile.fail())
				return false;

		}

		infile.close();

		return true;
	}
	else
		return false;
}

bool readSubstructuresTrackFile(const std::string& filename, const int n_frames, std::vector<std::string>& structure_names, std::vector<std::vector<subStructLabel_t>>& track)
{
	ifstream infile(filename.c_str());
	if (infile.is_open())
	{
		string dummy_string;
		// Skip the first, header line
		getline(infile,dummy_string);

		int n_structures;
		infile >> n_structures;
		if(infile.fail())
			return false;

		// Skip the rest of this line
		getline(infile,dummy_string);

		structure_names.resize(n_structures);

		track.resize(n_frames);
		for(vector<subStructLabel_t>& v : track)
			v.resize(n_structures);

		// Loop through the substructures
		for(int s = 0; s < n_structures; ++s)
		{
			// Read in the first line with name and number
			int dummy_int;
			infile >> dummy_int;
			if(infile.fail())
				return false;

			// Check the structure number matches what we expected
			if(dummy_int != s)
				return false;

			// Read in the name
			infile >> structure_names[s];
			if(infile.fail())
				return false;

			// Loop through frames
			getline(infile,dummy_string);
			int f = -1;
			for(string linestring; !getline(infile,linestring).eof() && !linestring.empty(); )
			{
				f++;
				stringstream ss(linestring);
				ss >> dummy_int;
				if(ss.fail())
					return false;

				if(dummy_int != f)
					return false;

				ss >> track[f][s].labelled;
				if(ss.fail())
					return false;
				ss >> track[f][s].present;
				if(ss.fail())
					return false;
				ss >> track[f][s].y;
				if(ss.fail())
					return false;
				ss >> track[f][s].x;
				if(ss.fail())
					return false;
				ss >> track[f][s].ori;
				if(ss.fail())
					return false;
			}
			// We do not have information on some of the frames at the end
			// Mark them as unlabelled
			while(f < n_frames - 1)
			{
				track[f][s].labelled = false;
				track[f][s].present = 0;
				track[f][s].y = -1;
				track[f][s].x = -1;
				track[f][s].ori = 0;
				f++;
			}
		}
		infile.close();
		return true;
	}
	else
		return false;
}

// Basically the same as above, except now we only want to return the structures that were
// asked for (in the given order)
bool readGivenSubstructures(const std::string& filename, const std::vector<std::string>& structs, const int n_frames, std::vector<std::vector<subStructLabel_t>>& track)
{
	// Read the whole substructures file first
	vector<string> read_struct_names;
	vector<vector<subStructLabel_t>> read_track;

	// Allocate space in the output array
	track.resize(n_frames);
	for(int f = 0; f < n_frames; ++f)
		track[f].resize(structs.size());

	if(!readSubstructuresTrackFile(filename, n_frames, read_struct_names, read_track))
		return false;

	// Now copy the relevant vectors into the output array
	for(unsigned s = 0; s < structs.size(); ++s)
	{
		// Find the location of this structure in the read list by searching for a matching name
		const auto match_it = std::find(read_struct_names.cbegin(),read_struct_names.cend(),structs[s]);
		if(match_it == read_struct_names.cend())
			// Means the name wasn't found
			return false;
		const int match_ind = std::distance(read_struct_names.cbegin(),match_it);

		// Copy the relevant information
		for(int f = 0; f < n_frames; ++f)
			track[f][s] = read_track[f][match_ind];
	}
	return true;
}

bool subStructureFileContains(const std::string& filename, const std::string& structname)
{
	ifstream infile(filename.c_str());
	if (infile.is_open())
	{
		for(string linestring; !getline(infile,linestring).eof(); )
		{
			stringstream ss(linestring);
			string dummy_string;
			ss >> dummy_string;
			ss >> dummy_string;

			if(structname == dummy_string)
			{
				for(string framestring; !getline(infile,framestring).eof() && !framestring.empty(); )
				{
					stringstream ss2(framestring);
					int dummy_int;
					int present;
					bool labelled;
					ss2 >> dummy_int;
					ss2 >> labelled;
					ss2 >> present;

					if(labelled && present == 1)
						return true;
				}
			}
		}
	}
	else
		cout << "Warning ignoring missing trackfile " << filename << endl;

	return false;
}

bool trackFileContainsView(const std::string& filename, const int view)
{
	ifstream infile(filename.c_str());
	if (infile.is_open())
	{
		std::string dummy_string;

		// Skip the first three lines
		getline(infile,dummy_string);
		getline(infile,dummy_string);
		getline(infile,dummy_string);

		// Loop through frames
		for(string linestring; !getline(infile,linestring).eof(); )
		{
			stringstream ss(linestring);

			ss >> dummy_string;
			bool present;
			ss >> present;
			int labelled;
			ss >> labelled;

			for(int i = 0; i < 3; ++i)
				ss >> dummy_string;

			int view_in;
			ss >> view_in;

			if(present && (labelled == 1) && (view == view_in))
				return true;
		}

		return false;
	}
	else
		cout << "Warning ignoring missing trackfile " << filename << endl;

	return false;
}

// Function to interpret the string defining the features
bool parseHoughPointFeatureDefinitionString(const std::string&feat_str, int& patchhalfsize, float& patch_radius_ratio)
{
	// A string stream for parsing the string
	std::stringstream ss;
	ss.str(feat_str);

	ss >> patchhalfsize;
	if(ss.fail())
		return false;

	ss >> patch_radius_ratio;
	if(ss.fail())
		return false;

	return true;
}

// Takes an 'img_to_valid' array, which for each 1D pixel index in an image lists
// either the id of that pixel in a valid list, or -1 if the point is not in the
// list. Transforms this such that all the -1s are replaced by -(n+1) , where n
// is the id of the closest point in the list.
// Uses the Manhatten distance metric for simplicity
void closestNeighboursInSet(std::vector<int>& img_to_valid, const int ysize, const int xsize)
{
	// Use the largest ID in the set +1 as a special value to indicate not yet
	// found
	const int neighbour_not_found = *(std::max_element(img_to_valid.cbegin(),img_to_valid.cend())) + 1;

	// Replace all negative values with this number
	std::replace_if(img_to_valid.begin(),img_to_valid.end(), [](int i){return i < 0;}, neighbour_not_found);

	// Initialise a vector containing distances
	std::vector<int> distances(img_to_valid.size());
	for(unsigned p = 0; p < img_to_valid.size(); ++p)
	{
		if(img_to_valid[p] == neighbour_not_found)
			distances[p] = -1;
		else
			distances[p] = 0;
	}

	// Temporary arrays to write to during iterations
	std::vector<int> distances_write(img_to_valid.size());
	std::vector<int> img_to_valid_write(img_to_valid.size());

	// Iterate until all of these have been replaced
	while(std::any_of(img_to_valid.cbegin(),img_to_valid.cend(), [=](int i){return i == neighbour_not_found;}))
	{
		// Initialise the temporary arrays to match the true arrays
		std::copy(distances.cbegin(),distances.cend(),distances_write.begin());
		std::copy(img_to_valid.cbegin(),img_to_valid.cend(),img_to_valid_write.begin());

		// Iterate over all points
		for(unsigned p = 0; p < img_to_valid.size(); ++p)
		{
			const int xpos = p%xsize;
			const int ypos = p/xsize;

			if(img_to_valid[p] == neighbour_not_found)
			{
				bool found = false;
				int best_distance = 0;
				int best_id = 0;

				// Check neighour above
				if(ypos > 0)
				{
					const int pp = p - xsize;
					if(img_to_valid[pp] != neighbour_not_found)
					{
						best_distance = distances[pp] + 1;
						best_id = img_to_valid[pp] >= 0 ? -(img_to_valid[pp]+1) : img_to_valid[pp];
						found = true;
					}
				}

				// Check neighour below
				if(ypos < ysize - 1)
				{
					const int pp = p + xsize;
					if(img_to_valid[pp] != neighbour_not_found)
					{
						const int candidate_distance = distances[pp] + 1;
						if(!found || candidate_distance < best_distance)
						{
							best_distance = candidate_distance;
							best_id = img_to_valid[pp] >= 0 ? -(img_to_valid[pp]+1) : img_to_valid[pp];
						}
						found = true;
					}
				}

				// Check left neighour
				if(xpos > 0)
				{
					const int pp = p - 1;
					if(img_to_valid[pp] != neighbour_not_found)
					{
						const int candidate_distance = distances[pp] + 1;
						if(!found || candidate_distance < best_distance)
						{
							best_distance = candidate_distance;
							best_id = img_to_valid[pp] >= 0 ? -(img_to_valid[pp]+1) : img_to_valid[pp];
						}
						found = true;
					}
				}

				// Check right neighour
				if(xpos < xsize - 1)
				{
					const int pp = p + 1;
					if(img_to_valid[pp] != neighbour_not_found)
					{
						const int candidate_distance = distances[pp] + 1;
						if(!found || candidate_distance < best_distance)
						{
							best_distance = candidate_distance;
							best_id = img_to_valid[pp] >= 0 ? -(img_to_valid[pp]+1) : img_to_valid[pp];
						}
						found = true;
					}
				}

				// If any of the neighbours had known distances, update the values in
				// the list
				if(found)
				{
					distances_write[p] = best_distance;
					img_to_valid_write[p] = best_id ;
				}
			}
		}

		// Copy the changes back to the true arrays
		std::copy(distances_write.cbegin(),distances_write.cend(),distances.begin());
		std::copy(img_to_valid_write.cbegin(),img_to_valid_write.cend(),img_to_valid.begin());
	}
}


// Function to read a mask file from a png image, check the size matches the expected size,
// shrink the mask, and resize if necessary
bool prepareMask(const std::string& filename, const cv::Size expected_size, cv::Mat_<unsigned char>& mask, const double shrink_distance, const cv::Size output_size, const int boundary)
{
	using namespace cv;

	// Read mask from file
	Mat raw_mask = imread(filename,cv::IMREAD_GRAYSCALE);

	if(raw_mask.empty())
	{
		std::cerr  << "ERROR: Could not open mask file " << filename << std::endl;
		return false;
	}
	if( (raw_mask.rows != expected_size.height) || (raw_mask.cols !=  expected_size.width) )
	{
		std::cerr  << "ERROR: Mask dimensions do not match video dimensions" << std::endl;
		return false;
	}

	// Use a distance transform to shrink the mask
	Mat shrunk_mask;
	if(shrink_distance > 0.0)
	{
		Mat distances;
		distanceTransform(raw_mask,distances,cv::DIST_L2,5);

		// Threshold this
		threshold(distances,shrunk_mask,shrink_distance,1,THRESH_BINARY);
	}
	else
		shrunk_mask = raw_mask;

	// Resize the mask if necessary
	if(output_size.width > 0)
		resize(shrunk_mask,shrunk_mask,output_size);

	// Change to unsigned char for return
	shrunk_mask.convertTo(mask,CV_8U);

	// Ensure a boundary of a certain width
	if(boundary > 0)
	{
		Mat roi;
		// First do the 'left' rectangle
		roi = mask(cv::Range::all(),cv::Range(0,boundary));
		roi = cv::Scalar(0);
		// 'Right' rectangle
		roi = mask(cv::Range::all(),cv::Range(mask.cols-boundary,mask.cols));
		roi = cv::Scalar(0);
		// 'Top' rectangle
		roi = mask(cv::Range(0,boundary),cv::Range::all());
		roi = cv::Scalar(0);
		// 'Bottom'
		roi = mask(cv::Range(mask.rows-boundary,mask.rows),cv::Range::all());
		roi = cv::Scalar(0);
	}

	return true;

}

// Function to find a list of valid pixel indices from a mask
// Additional options include ensuring that all pixels are at least "boundary" pixels from the image edge
// only selecting pixels at a spacing of "stride", and finding a vector that contains the "reverse" translation -
// i.e. for each image location returns its index in the valid pixels list
void findValidPixels(std::vector<cv::Point>& pixels, const cv::Mat_<unsigned char>& mask, const int stride, const int boundary, std::vector<int>* const reverse)
{
	const int xsize = mask.cols;
	const int ysize = mask.rows;

	// Clear the vector and reserve enough space
	pixels.clear();
	pixels.reserve(xsize*ysize/stride);

	if(reverse != nullptr)
	{
		reverse->clear();
		reverse->reserve(xsize*ysize);
	}

	// Loop through pixels
	for(int y = 0; y < ysize; ++y)
	{
		for(int x = 0; x < xsize; ++x)
		{
			if( (x > boundary) && (y > boundary) && (x < (xsize - boundary)) && (y < (ysize - boundary)) && (mask(y,x) > 0) && (x % stride == 0) && (y % stride == 0) )
			{
				if(reverse != nullptr)
					reverse->emplace_back(pixels.size());
				pixels.emplace_back(cv::Point(x,y));
			}
			else if(reverse != nullptr)
				reverse->emplace_back(-1);
		}
	}

	// Undo unnecessary reserving
	pixels.shrink_to_fit();

	// Do the distance transform to find closest points for areas not in the set
	if(reverse != nullptr)
	{
		std::vector<int>& ref = *reverse;
		closestNeighboursInSet(ref,ysize,xsize);
	}
}

// Check that the features described by the feat_str match those in the other input parameters, except for Jmax, which is allowed to vary
// and is returned by reference
bool checkFeaturesMatch(const std::string& feat_str, std::vector<int>& new_Jmax, const int train_radius, const std::vector<int>& J, const std::vector<int>& K,
						const std::vector<int>& M, const std::vector<int>& max_rot_order, const std::vector<int>& coupling_type,
						const std::vector<int>& basis_type, const std::vector<float>& wl, const std::vector<featType_t>& feat_type)
{
	int new_train_radius;
	vector<int> new_J,new_K,new_M,new_max_rot_order,new_coupling_type,new_basis_type;
	vector<float> new_wl;
	vector<string> new_feat_type_str;
	vector<thesisUtilities::featType_t> new_feat_type;
	if(!parseFeatureDefinitionString(feat_str, new_J, new_K, new_M, new_max_rot_order, new_feat_type, new_wl, new_coupling_type,new_basis_type, new_Jmax, new_train_radius))
	{
		cerr << "ERROR: problem reading features from model file, the string was " << feat_str << endl;
		return false;
	}
	// Check that values match the other feature sets
	if(		(J.size() != new_J.size()) ||
			(new_train_radius != train_radius) ||
			!std::equal(J.cbegin(),J.cend(),new_J.cbegin()) ||
			!std::equal(K.cbegin(),K.cend(),new_K.cbegin()) ||
			!std::equal(M.cbegin(),M.cend(),new_M.cbegin()) ||
			!std::equal(wl.cbegin(),wl.cend(),new_wl.cbegin()) ||
			!std::equal(max_rot_order.cbegin(),max_rot_order.cend(),new_max_rot_order.cbegin()) ||
			!std::equal(coupling_type.cbegin(),coupling_type.cend(),new_coupling_type.cbegin()) ||
			!std::equal(basis_type.cbegin(),basis_type.cend(),new_basis_type.cbegin()) ||
			!std::equal(feat_type.cbegin(),feat_type.cend(),new_feat_type.cbegin())
	  )
	{
		return false;
	}
	return true;
}


// Check that the features described by the feat_str match those in the other input parameters, except for Jmax, which is allowed to vary
// and is returned by reference
bool checkHoughFeaturesMatch(const std::string& feat_str, std::vector<int>& new_Jmax, const int train_radius, const std::vector<int>& J, const std::vector<int>& K,
						const std::vector<int>& M, const std::vector<int>& max_rot_order, const std::vector<int>& coupling_type,
						const std::vector<int>& basis_type, const std::vector<float>& wl, const std::vector<featType_t>& feat_type)
{
	int new_train_radius;
	float new_patch_radius_ratio;
	vector<int> new_J,new_K,new_M,new_max_rot_order,new_coupling_type,new_basis_type;
	vector<float> new_wl;
	vector<string> new_feat_type_str;
	vector<thesisUtilities::featType_t> new_feat_type;
	if(!thesisUtilities::parseRIHoughFeatureDefinitionString(feat_str, new_J, new_K, new_M, new_max_rot_order, new_feat_type, new_wl, new_coupling_type,new_basis_type, new_Jmax, new_train_radius, new_patch_radius_ratio))
	{
		cerr << "ERROR: problem reading features from model file, the string was " << feat_str << endl;
		return false;
	}
	// Check that values match the other feature sets
	if(		(J.size() != new_J.size()) ||
			(new_train_radius != train_radius) ||
			!std::equal(J.cbegin(),J.cend(),new_J.cbegin()) ||
			!std::equal(K.cbegin(),K.cend(),new_K.cbegin()) ||
			!std::equal(M.cbegin(),M.cend(),new_M.cbegin()) ||
			!std::equal(wl.cbegin(),wl.cend(),new_wl.cbegin()) ||
			!std::equal(max_rot_order.cbegin(),max_rot_order.cend(),new_max_rot_order.cbegin()) ||
			!std::equal(coupling_type.cbegin(),coupling_type.cend(),new_coupling_type.cbegin()) ||
			!std::equal(basis_type.cbegin(),basis_type.cend(),new_basis_type.cbegin()) ||
			!std::equal(feat_type.cbegin(),feat_type.cend(),new_feat_type.cbegin())
	  )
	{

		cerr << "ERROR: features in the following hough model do not match those in the base model:" << endl;
		return false;
	}
	if(new_patch_radius_ratio >= 0.0)
	{
		cerr << "ERROR: hough model is not using heart scaling in the following model:" << endl;
		return false;
	}
	return true;
}

} // end of namespace
