
// Default constructor
template<int TNClasses>
particleFilterPosClassOri<TNClasses>::particleFilterPosClassOri()
: particleFilterPosClassOri(0,0,0,0.0,"",nullptr)
{

}

// Full constructor
template<int TNClasses>
particleFilterPosClassOri<TNClasses>::particleFilterPosClassOri(const int ysize, const int xsize, size_t n_particles, const double radius, const std::string& def_file, cv::Mat_<unsigned char> * const valid_mask)
: particleFilterBase< stateModelOriAwarePosClass<TNClasses,1> ,
					  stateModelOri<TNClasses,0> > (ysize, xsize, n_particles)
{
	std::get<0>(this->state_models) = stateModelOriAwarePosClass<TNClasses,1>(ysize,xsize,def_file,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();

	std::get<1>(this->state_models) = stateModelOri<TNClasses,0> (ysize,xsize,def_file,radius);
	this->init &= std::get<1>(this->state_models).initialise();
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses>
void particleFilterPosClassOri<TNClasses>::visualiseOri(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClass& s_pos = std::get<0>(this->particles[p]);
		const stateOri& s_ori = std::get<1>(this->particles[p]);
		const int y = std::round(s_pos.y*(float(disp->rows)/float(this->ysize)));
		const int x = std::round(s_pos.x*(float(disp->cols)/float(this->xsize)));
		cv::Scalar colour(0,0,255);
		if(!s_pos.visible)
		{
			colour /= 2;
		}
		cv::arrowedLine(disp[s_pos.c-1],cv::Point(x,y),cv::Point(x+5.0*std::cos(s_ori.ori),y-5.0*std::sin(s_ori.ori)),cv::Scalar(0,0,255),1,8,0,0.5);
	}
}
