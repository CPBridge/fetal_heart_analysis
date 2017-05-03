
// Default constructor
template<int TNClasses>
particleFilterPosClassJoinedOri<TNClasses>::particleFilterPosClassJoinedOri()
: particleFilterPosClassJoinedOri(0,0,0,0.0,"",nullptr)
{

}

// Full constructor
template<int TNClasses>
particleFilterPosClassJoinedOri<TNClasses>::particleFilterPosClassJoinedOri(const int ysize, const int xsize, size_t n_particles, const double radius, const std::string& def_file, cv::Mat_<unsigned char> * const valid_mask)
: particleFilterBase< stateModelPosClassOri<TNClasses> > (ysize, xsize, n_particles)
{
	std::get<0>(this->state_models) = stateModelPosClassOri<TNClasses>(ysize,xsize,def_file,radius,valid_mask);
	this->init = std::get<0>(this->state_models).initialise();
}

// Add arrows to an existing image to show the particles' location and orientation
template<int TNClasses>
void particleFilterPosClassJoinedOri<TNClasses>::visualiseOri(cv::Mat_<cv::Vec3b>* const disp) const
{
	// Loop over all particles
	for(unsigned p = 0; p < this->n_particles; ++p)
	{
		const statePosClassOri& s_pos = std::get<0>(this->particles[p]);
		const int y = std::round(s_pos.y*(float(disp->rows)/float(this->ysize)));
		const int x = std::round(s_pos.x*(float(disp->cols)/float(this->xsize)));
		cv::Scalar colour(0,0,255);
		if(!s_pos.visible)
		{
			colour /= 2;
		}
		cv::arrowedLine(disp[s_pos.c-1],cv::Point(x,y),cv::Point(x+5.0*std::cos(s_pos.ori),y-5.0*std::sin(s_pos.ori)),cv::Scalar(0,0,255),1,8,0,0.5);
	}
}
