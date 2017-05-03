#include <fstream>
#include "displayFunctions.h"

// Default constructor
template<int TNClasses, int TNStructs>
particleFilterSingleStructs<TNClasses,TNStructs>::particleFilterSingleStructs()
: particleFilterSingleStructs(0,0,0,0.0,0.0,"","","",std::vector<std::string>(),nullptr,nullptr)
{

}

// Full constructor
template<int TNClasses, int TNStructs>
particleFilterSingleStructs<TNClasses,TNStructs>::particleFilterSingleStructs(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
																const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs, const std::vector<std::string>& subs_names,
																cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask)
// Base class initialiser
: particleFilterSingleStructsBaseClass<TNClasses,TNStructs>(ysize, xsize, n_particles)

{
	// Initialise the position state model
	std::get<0>(this->state_models) = stateModelOriAwarePosClass<TNClasses,2>(ysize,xsize,def_file_posclass,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();

	// Initialise the phase state model
	std::get<1>(this->state_models) = stateModelPhase<TNClasses,0>(ysize,xsize,def_file_phase, radius, frame_rate);
	this->init &= std::get<1>(this->state_models).initialise();

	// Initialise the orientation state model
	std::get<2>(this->state_models) = stateModelOri<TNClasses,0> (ysize,xsize,def_file_posclass,radius);
	this->init &= std::get<2>(this->state_models).initialise();

	// Kick off recursive initialisation of the structure state models
	this->init &= initialiseStructModel(int2type<TNStructs-1>(),ysize,xsize, radius, def_file_substructs, subs_names, subs_valid_mask);

	// Read the file to get an easily accesible list of which structures are in which view
	if(this->init)
		getStructureInViewList(def_file_substructs);
}

// Recursive method to initialise the I'th structure model
template<int TNClasses, int TNStructs>
template <size_t I>
bool particleFilterSingleStructs<TNClasses,TNStructs>::initialiseStructModel(int2type<I>, const int ysize, const int xsize,
																				const double radius, const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask)
{
	// Initialise this model
	std::get<I+3>(this->state_models) = stateModelSingleStructure<TNClasses,0,1,2>(ysize, xsize, I, def_file_substructs, radius, subs_names, subs_valid_mask);
	const bool init = std::get<I+3>(this->state_models).initialise();

	// Recursive call to the next
	return init && initialiseStructModel(int2type<I-1>(),ysize,xsize, radius, def_file_substructs, subs_names, subs_valid_mask);
}

// Base case of the above recursion
template<int TNClasses, int TNStructs>
bool particleFilterSingleStructs<TNClasses,TNStructs>::initialiseStructModel(int2type<0>, const int ysize, const int xsize,
																				const double radius, const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask)
{
	// Initialise the final model
	std::get<3>(this->state_models) = stateModelSingleStructure<TNClasses,0,1,2>(ysize, xsize, 0, def_file_substructs, radius, subs_names, subs_valid_mask);
	return std::get<3>(this->state_models).initialise();
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses, int TNStructs>
void particleFilterSingleStructs<TNClasses,TNStructs>::visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClass& s_pos = std::get<0>(this->particles[p]);
		const statePhase& s_ph = std::get<1>(this->particles[p]);
		const stateOri& s_ori = std::get<2>(this->particles[p]);
		const int y = round(s_pos.y*(float(disp->rows)/float(this->ysize)));
		const int x = round(s_pos.x*(float(disp->cols)/float(this->xsize)));
		unsigned char R,G,B;
		thesisUtilities::ang2rgb(s_ph.ph,R,G,B);
		cv::arrowedLine(disp[s_pos.c-1],cv::Point(x,y),cv::Point(x+5.0*std::cos(s_ori.ori),y-5.0*std::sin(s_ori.ori)),cv::Scalar(B,G,R),1,8,0,0.5);
	}
}

// Get an output image that visualises the positions of all the particles
template<int TNClasses, int TNStructs>
void particleFilterSingleStructs<TNClasses,TNStructs>::visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const
{
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		// Get the substructure positions in array form
		std::array<double,TNStructs> subs_x, subs_y;
		std::array<structVisible_enum,TNStructs> subs_visible;
		structPositionArray(this->particles[p],subs_y,subs_x,subs_visible);

		for(unsigned s = 0; s < TNStructs; ++s)
		{
			const int y = round(subs_y[s]*(float(disp->rows)/float(this->ysize)));
			const int x = round(subs_x[s]*(float(disp->cols)/float(this->xsize)));

			if((y > 0) && (y < disp->rows) && (x > 0) && (x < disp->cols))
			{
				if( subs_visible[s] == structVisible_enum::svVisible)
					disp[s](y,x) = cv::Vec3b(0,255,0);
				else if ( subs_visible[s] == structVisible_enum::svHidden || subs_visible[s] == structVisible_enum::svHiddenOffEdge)
					disp[s](y,x) = cv::Vec3b(0,0,255);
			}
		}
	}
}

// Take a combined state tuple and extract the structure positions into a std::array
// to make it much easier to use from external functions
template<int TNClasses, int TNStructs>
void particleFilterSingleStructs<TNClasses,TNStructs>::structPositionArray(const combined_state_type& s, std::array<double,TNStructs>& y_arr, std::array<double,TNStructs>& x_arr, std::array<structVisible_enum,TNStructs>& visible_arr) const
{
	// Kick off the recursion
	structPositionArray_impl(int2type<TNStructs-1>(),s,y_arr,x_arr,visible_arr);
}

// Recursive function to extract tuple elements into array
template<int TNClasses, int TNStructs>
template <size_t I>
void particleFilterSingleStructs<TNClasses,TNStructs>::structPositionArray_impl(int2type<I>, const combined_state_type& s, std::array<double,TNStructs>& y_arr, std::array<double,TNStructs>& x_arr, std::array<structVisible_enum,TNStructs>& visible_arr) const
{
	// Extract the x and y for this structure into the array
	y_arr[I] = std::get<I+3>(s).y;
	x_arr[I] = std::get<I+3>(s).x;
	visible_arr[I] = std::get<I+3>(s).visible;

	// Recurse
	structPositionArray_impl(int2type<I-1>(),s,y_arr,x_arr,visible_arr);
}

// Base case - end the recursion
template<int TNClasses, int TNStructs>
void particleFilterSingleStructs<TNClasses,TNStructs>::structPositionArray_impl(int2type<0>, const combined_state_type& s, std::array<double, TNStructs>& y_arr, std::array<double, TNStructs>& x_arr, std::array<structVisible_enum,TNStructs>& visible_arr) const
{
	// Extract the x and y for this structure into the array
	y_arr[0] = std::get<3>(s).y;
	x_arr[0] = std::get<3>(s).x;
	visible_arr[0] = std::get<3>(s).visible;

	// No recursion
}

template<int TNClasses, int TNStructs>
void particleFilterSingleStructs<TNClasses,TNStructs>::getStructureInViewList(const std::string& def_file)
{
	// Open the file
	std::ifstream infile(def_file.c_str());

	std::string dummy_string;

	// Skip lines to get to the structures list
	for(int i = 0; i < 28; ++i)
		getline(infile,dummy_string);

	for(int s = 0; s < TNStructs; ++s)
	{
		// Set all to false intially for this structure
		for(int c = 0; c < TNClasses; ++c)
			structure_in_view[s][c] = false;

		infile >> dummy_string; // name
		getline(infile,dummy_string);
		std::stringstream ss(dummy_string);
		int tempint;
		while(ss >> tempint)
			structure_in_view[s][tempint-1] = true;
	}
}
