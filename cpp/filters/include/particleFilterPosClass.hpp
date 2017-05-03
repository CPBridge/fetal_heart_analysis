#ifndef PARTICLEFILTERPOSCLASS_H
#define PARTICLEFILTERPOSCLASS_H

#include "stateModelPosClass.hpp"
#include "particleFilterBase.hpp"

template<int TNClasses>
class particleFilterPosClass : public particleFilterBase<stateModelPosClass<TNClasses>>
{
	public:
		// Constructors
		// ------------
		particleFilterPosClass();
		particleFilterPosClass(const int ysize, const int xsize, size_t n_particles, const double radius, const std::string& def_file, cv::Mat_<unsigned char> * const valid_mask);

		// Methods
		// -------
		void visualise(cv::Mat_<float>* const disp) const;
		void visualiseHidden(cv::Mat_<cv::Vec3b>* const disp) const;

};

#include "particleFilterPosClass.tpp"

#endif
