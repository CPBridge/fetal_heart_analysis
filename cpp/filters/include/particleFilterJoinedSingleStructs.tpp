#include "displayFunctions.h"


// Default constructor
template<int TNClasses, int TNStructs>
particleFilterJoinedSingleStructs<TNClasses,TNStructs>::particleFilterJoinedSingleStructs()
: particleFilterJoinedSingleStructs(0,0,0,0.0,0.0,"","","",std::vector<std::string>(),nullptr,nullptr)
{

}

// Full constructor
template<int TNClasses, int TNStructs>
particleFilterJoinedSingleStructs<TNClasses,TNStructs>::particleFilterJoinedSingleStructs(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
																const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs, const std::vector<std::string>& subs_names,
																cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask)
// Base class initialiser
: particleFilterJoinedSingleStructsBaseClass<TNClasses,TNStructs>(ysize, xsize, n_particles)

{
	// Initialise the position state model
	std::get<0>(this->state_models) = stateModelPosClassOri<TNClasses>(ysize,xsize,def_file_posclass,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();

	// Initialise the phase state model
	std::get<1>(this->state_models) = stateModelPhase<TNClasses,0>(ysize,xsize,def_file_phase, radius, frame_rate);
	this->init &= std::get<1>(this->state_models).initialise();

	// Kick off recursive initialisation of the structure state models
	this->init &= initialiseStructModel(int2type<TNStructs-1>(),ysize,xsize, radius, def_file_substructs, subs_names, subs_valid_mask);
}

// Recursive method to initialise the I'th structure model
template<int TNClasses, int TNStructs>
template <size_t I>
bool particleFilterJoinedSingleStructs<TNClasses,TNStructs>::initialiseStructModel(int2type<I>, const int ysize, const int xsize,
																				const double radius, const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask)
{
	// Initialise this model
	std::get<I+2>(this->state_models) = stateModelSingleStructure<TNClasses,0,1,0>(ysize, xsize, I, def_file_substructs, radius, subs_names, subs_valid_mask);
	const bool init = std::get<I+2>(this->state_models).initialise();

	// Recursive call to the next
	return init && initialiseStructModel(int2type<I-1>(),ysize,xsize, radius, def_file_substructs, subs_names, subs_valid_mask);
}

// Base case of the above recursion
template<int TNClasses, int TNStructs>
bool particleFilterJoinedSingleStructs<TNClasses,TNStructs>::initialiseStructModel(int2type<0>, const int ysize, const int xsize,
																				const double radius, const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask)
{
	// Initialise the final model
	std::get<2>(this->state_models) = stateModelSingleStructure<TNClasses,0,1,0>(ysize, xsize, 0, def_file_substructs, radius, subs_names, subs_valid_mask);
	return std::get<2>(this->state_models).initialise();
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses, int TNStructs>
void particleFilterJoinedSingleStructs<TNClasses,TNStructs>::visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClassOri& s_pos = std::get<0>(this->particles[p]);
		const statePhase& s_ph = std::get<1>(this->particles[p]);
		const int y = round(s_pos.y*(float(disp->rows)/float(this->ysize)));
		const int x = round(s_pos.x*(float(disp->cols)/float(this->xsize)));
		unsigned char R,G,B;
		thesisUtilities::ang2rgb(s_ph.ph,R,G,B);
		cv::arrowedLine(disp[s_pos.c-1],cv::Point(x,y),cv::Point(x+5.0*std::cos(s_pos.ori),y-5.0*std::sin(s_pos.ori)),cv::Scalar(B,G,R),1,8,0,0.5);
	}
}

// Get an output image that visualises the positions of all the particles
template<int TNClasses, int TNStructs>
void particleFilterJoinedSingleStructs<TNClasses,TNStructs>::visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const
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

			if( subs_visible[s] == structVisible_enum::svVisible) //&& (subs_y[s] > 0.0) && (subs_y[s] < this->ysize) && (subs_x[s] > 0.0) && (subs_x[s] < this->xsize))
				//disp[s](std::floor(subs_y[s]),std::floor(subs_x[s])) += this->w[p]*this->n_particles;
					disp[s](y,x) = cv::Vec3b(0,255,0);
			else if ( subs_visible[s] == structVisible_enum::svHidden || subs_visible[s] == structVisible_enum::svHiddenOffEdge)
				disp[s](y,x) = cv::Vec3b(0,0,255);
		}
	}
}

// Take a combined state tuple and extract the structure positions into a std::array
// to make it much easier to use from external functions
template<int TNClasses, int TNStructs>
void particleFilterJoinedSingleStructs<TNClasses,TNStructs>::structPositionArray(const combined_state_type& s, std::array<double,TNStructs>& y_arr, std::array<double,TNStructs>& x_arr, std::array<structVisible_enum,TNStructs>& visible_arr) const
{
	// Kick off the recursion
	structPositionArray_impl(int2type<TNStructs-1>(),s,y_arr,x_arr,visible_arr);
}

// Recursive function to extract tuple elements into array
template<int TNClasses, int TNStructs>
template <size_t I>
void particleFilterJoinedSingleStructs<TNClasses,TNStructs>::structPositionArray_impl(int2type<I>, const combined_state_type& s, std::array<double,TNStructs>& y_arr, std::array<double,TNStructs>& x_arr, std::array<structVisible_enum,TNStructs>& visible_arr) const
{
	// Extract the x and y for this structure into the array
	y_arr[I] = std::get<I+2>(s).y;
	x_arr[I] = std::get<I+2>(s).x;
	visible_arr[I] = std::get<I+2>(s).visible;

	// Recurse
	structPositionArray_impl(int2type<I-1>(),s,y_arr,x_arr,visible_arr);
}

// Base case - end the recursion
template<int TNClasses, int TNStructs>
void particleFilterJoinedSingleStructs<TNClasses,TNStructs>::structPositionArray_impl(int2type<0>, const combined_state_type& s, std::array<double, TNStructs>& y_arr, std::array<double, TNStructs>& x_arr, std::array<structVisible_enum,TNStructs>& visible_arr) const
{
	// Extract the x and y for this structure into the array
	y_arr[0] = std::get<2>(s).y;
	x_arr[0] = std::get<2>(s).x;
	visible_arr[0] = std::get<2>(s).visible;

	// No recursion
}
