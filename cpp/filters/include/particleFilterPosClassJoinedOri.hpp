#ifndef PARTICLEFILTERPOSCLASSJOINEDORI_HPP
#define PARTICLEFILTERPOSCLASSJOINEDORI_HPP

#include <opencv2/core/core.hpp>
#include "stateModelPosClassOri.hpp"
#include "particleFilterBase.hpp"

template<int TNClasses>
class particleFilterPosClassJoinedOri : public particleFilterBase < stateModelPosClassOri<TNClasses> >
{
	public:
		// Constructors
		// ------------
		particleFilterPosClassJoinedOri();
		particleFilterPosClassJoinedOri(const int ysize, const int xsize, size_t n_particles, const double radius, const std::string& def_file, cv::Mat_<unsigned char> * const valid_mask);

		// Methods
		// -------
		void visualiseOri(cv::Mat_<cv::Vec3b>* const disp) const;

	protected:

};

#include "particleFilterPosClassJoinedOri.tpp"

#endif
