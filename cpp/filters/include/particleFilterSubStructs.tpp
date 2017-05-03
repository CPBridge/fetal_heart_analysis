#include "displayFunctions.h"


// Default constructor
template<int TNClasses>
particleFilterSubStructs<TNClasses>::particleFilterSubStructs()
: particleFilterSubStructs(0,0,0,0.0,0.0,"","","",std::vector<std::string>(),nullptr,nullptr)
{

}

// Full constructor
template<int TNClasses>
particleFilterSubStructs<TNClasses>::particleFilterSubStructs(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
																const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs,
																const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask)
// Base class initialiser
: particleFilterBase < stateModelOriAwarePosClass<TNClasses,2> ,
					   stateModelPhase<TNClasses,0> ,
					   stateModelOri<TNClasses,0> ,
					   stateModelSubstructuresPCA<TNClasses,0,1,2>
					 > (ysize, xsize, n_particles)

{
	std::get<0>(this->state_models) = stateModelOriAwarePosClass<TNClasses,2>(ysize,xsize,def_file_posclass,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();

	// Initialise the phase state model
	std::get<1>(this->state_models) = stateModelPhase<TNClasses,0>(ysize,xsize,def_file_phase, radius, frame_rate);
	this->init &= std::get<1>(this->state_models).initialise();

	std::get<2>(this->state_models) = stateModelOri<TNClasses,0> (ysize,xsize,def_file_posclass,radius);
	this->init &= std::get<2>(this->state_models).initialise();

	std::get<3>(this->state_models) = stateModelSubstructuresPCA<TNClasses,0,1,2> (ysize,xsize,def_file_substructs,radius,subs_names,subs_valid_mask);
	this->init &= std::get<3>(this->state_models).initialise();
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses>
void particleFilterSubStructs<TNClasses>::visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClass& s_pos = std::get<0>(this->particles[p]);
		const statePhase& s_ph = std::get<1>(this->particles[p]);
		const stateOri& s_ori = std::get<2>(this->particles[p]);
		const int y = std::round(s_pos.y*(float(disp->rows)/float(this->ysize)));
		const int x = std::round(s_pos.x*(float(disp->cols)/float(this->xsize)));
		unsigned char R,G,B;
		thesisUtilities::ang2rgb(s_ph.ph,R,G,B);
		cv::arrowedLine(disp[s_pos.c-1],cv::Point(x,y),cv::Point(x+5.0*std::cos(s_ori.ori),y-5.0*std::sin(s_ori.ori)),cv::Scalar(B,G,R),1,8,0,0.5);
	}
}

// Get an output image that visualises the positions of all the particles
template<int TNClasses>
void particleFilterSubStructs<TNClasses>::visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const
{
	const int n_structures = std::get<3>(this->state_models).getNumStructures();
	for(int s = 0 ; s < n_structures; ++s)
	{
		for(unsigned p = 0; p < this->n_particles; ++p)
		{
			const int heart_class = std::get<0>(this->particles[p]).c;
			if(std::get<3>(this->state_models).structInView(s,heart_class-1))
			{
				const int x = std::round(std::get<3>(this->particles[p]).x[s]*(float(disp->rows)/float(this->ysize)));
				const int y = std::round(std::get<3>(this->particles[p]).y[s]*(float(disp->rows)/float(this->ysize)));
				const structVisible_enum visible = std::get<3>(this->particles[p]).visible[s];

				if((y > 0) && (y < disp->rows) && (x > 0) && (x < disp->cols))
				{
					if(visible == structVisible_enum::svVisible)
						disp[s](y,x) = cv::Vec3b(0,255,0);
					else
						disp[s](y,x) = cv::Vec3b(0,0,255);
				}
			}
		}
	}
}
