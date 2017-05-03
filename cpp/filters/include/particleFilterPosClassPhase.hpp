#ifndef PARTICLEFILTERPOSCLASSPHASE_HPP
#define PARTICLEFILTERPOSCLASSPHASE_HPP

#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelPosClass.hpp"
#include "stateModelPhase.hpp"

template<int TNClasses>
class particleFilterPosClassPhase : public particleFilterBase< stateModelPosClass<TNClasses> , stateModelPhase<TNClasses,0> >
{
	public:
		// Constructors
		// ------------
		particleFilterPosClassPhase();
		particleFilterPosClassPhase(const int ysize, const int xsize, size_t n_particles, const double radius, const double frame_rate,
									const std::string& def_file_posclass, const std::string& def_file_phase, cv::Mat_<unsigned char> * const valid_mask);

		// Methods
		// -------
		void visualisePhase(cv::Mat_<cv::Vec3b>* const disp) const;

	protected:

};

#include "particleFilterPosClassPhase.tpp"

#endif
