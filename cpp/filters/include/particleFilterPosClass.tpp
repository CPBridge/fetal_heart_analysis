
// Default constructor
template<int TNClasses>
particleFilterPosClass<TNClasses>::particleFilterPosClass()
: particleFilterPosClass(0,0,0,0.0,"",nullptr)
{

}

// Full constructor
template<int TNClasses>
particleFilterPosClass<TNClasses>::particleFilterPosClass(const int ysize, const int xsize, size_t n_particles, const double radius, const std::string& def_file, cv::Mat_<unsigned char> * const valid_mask)
: particleFilterBase<stateModelPosClass<TNClasses>>(ysize, xsize, n_particles)
{
	std::get<0>(this->state_models) = stateModelPosClass<TNClasses>(ysize,xsize,def_file,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();
}


// Get an output image that visualises the positions of all the particles
template<int TNClasses>
void particleFilterPosClass<TNClasses>::visualise(cv::Mat_<float>* const disp) const
{
	for(int c = 0 ; c < TNClasses; ++c)
		disp[c] = cv::Mat::zeros(this->ysize,this->xsize,CV_8U);

	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClass& s = std::get<0>(this->particles[p]);
		disp[s.c-1](std::floor(s.y),std::floor(s.x)) += this->w[p]*this->n_particles;
	}
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses>
void particleFilterPosClass<TNClasses>::visualiseHidden(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClass& s_pos = std::get<0>(this->particles[p]);
		const int y = std::round(s_pos.y*(float(disp->rows)/float(this->ysize)));
		const int x = std::round(s_pos.x*(float(disp->cols)/float(this->xsize)));
		if(s_pos.visible)
			disp[s_pos.c-1](y,x) = cv::Vec3b(0,255,0);
		else
			disp[s_pos.c-1](y,x) = cv::Vec3b(0,0,255);
	}
}
