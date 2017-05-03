#include "displayFunctions.h" /* ang2rgb */

template <int TNClasses>
particleFilterPosClassPhase<TNClasses>::particleFilterPosClassPhase()
: particleFilterPosClassPhase(0,0,0,0.0,0.0,"","",nullptr)
{

}

template <int TNClasses>
particleFilterPosClassPhase<TNClasses>::particleFilterPosClassPhase(const int ysize, const int xsize, size_t n_particles,
							const double radius, const double frame_rate, const std::string& def_file_posclass,
							const std::string& def_file_phase, cv::Mat_<unsigned char> * const valid_mask)
: particleFilterBase< stateModelPosClass<TNClasses> , stateModelPhase<TNClasses,0> >(ysize, xsize, n_particles)
{
	// Initialise the position class state model
	std::get<0>(this->state_models) = stateModelPosClass<TNClasses>(ysize,xsize,def_file_posclass,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();

	// Initialise the phase state model
	std::get<1>(this->state_models) = stateModelPhase<TNClasses,0>(ysize,xsize,def_file_phase,radius, frame_rate);
	this->init &= std::get<1>(this->state_models).initialise();
}

template <int TNClasses>
void particleFilterPosClassPhase<TNClasses>::visualisePhase(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClass& s_posclass = std::get<0>(this->particles[p]);
		const statePhase& s_phase = std::get<1>(this->particles[p]);
		unsigned char R,G,B;
		thesisUtilities::ang2rgb(s_phase.ph,R,G,B);
		const int y = s_posclass.y*(float(disp[s_posclass.c-1].rows)/float(this->ysize));
		const int x = s_posclass.x*(float(disp[s_posclass.c-1].cols)/float(this->xsize));

		if(!s_posclass.visible)
		{
			R/=2;
			G/=2;
			B/=2;
		}
		disp[s_posclass.c-1](y,x) = cv::Vec3b(B,G,R);
	}

}
