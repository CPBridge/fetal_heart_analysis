#ifndef PARTICLEFILTERPOSCLASSORI_HPP
#define PARTICLEFILTERPOSCLASSORI_HPP

#include <opencv2/core/core.hpp>
#include "stateModelOriAwarePosClass.hpp"
#include "stateModelOri.hpp"
#include "particleFilterBase.hpp"

template<int TNClasses>
class particleFilterPosClassOri : public particleFilterBase < stateModelOriAwarePosClass<TNClasses,1> ,
															  stateModelOri<TNClasses,0> >
{
	public:
		// Constructors
		// ------------
		particleFilterPosClassOri();
		particleFilterPosClassOri(const int ysize, const int xsize, size_t n_particles, const double radius, const std::string& def_file, cv::Mat_<unsigned char> * const valid_mask);

		// Methods
		// -------
		void visualiseOri(cv::Mat_<cv::Vec3b>* const disp) const;

	protected:

};

#include "particleFilterPosClassOri.tpp"

#endif
