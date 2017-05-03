#include "displayFunctions.h"


// Default constructor
template<int TNClasses>
particleFilterPosClassJoinedOriPhase<TNClasses>::particleFilterPosClassJoinedOriPhase()
: particleFilterPosClassJoinedOriPhase(0,0,0,0.0,0.0,"","",nullptr)
{

}

// Full constructor
template<int TNClasses>
particleFilterPosClassJoinedOriPhase<TNClasses>::particleFilterPosClassJoinedOriPhase(const int ysize, const int xsize, size_t n_particles, const double radius,
																			const double frame_rate, const std::string& def_file_posclass,
																			const std::string& def_file_phase, cv::Mat_<unsigned char>* const valid_mask)
: particleFilterBase < stateModelPosClassOri<TNClasses> , stateModelPhase<TNClasses,0>  > (ysize, xsize, n_particles)
{
	std::get<0>(this->state_models) = stateModelPosClassOri<TNClasses>(ysize,xsize,def_file_posclass,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();

	// Initialise the phase state model
	std::get<1>(this->state_models) = stateModelPhase<TNClasses,0>(ysize,xsize,def_file_phase, radius, frame_rate);
	this->init &= std::get<1>(this->state_models).initialise();
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses>
void particleFilterPosClassJoinedOriPhase<TNClasses>::visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const
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
		if(!s_pos.visible)
		{
			R/=2;
			G/=2;
			B/=2;
		}
		arrowedLine(disp[s_pos.c-1],cv::Point(x,y),cv::Point(x+5.0*std::cos(s_pos.ori),y-5.0*std::sin(s_pos.ori)),cv::Scalar(B,G,R),1,8,0,0.5);
	}
}
