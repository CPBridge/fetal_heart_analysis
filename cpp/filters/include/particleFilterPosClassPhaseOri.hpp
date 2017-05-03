#ifndef PARTICLEFILTERPOSCLASSPHASEORI_HPP
#define PARTICLEFILTERPOSCLASSPHASEORI_HPP

#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelOriAwarePosClass.hpp"
#include "stateModelOri.hpp"
#include "stateModelPhase.hpp"


template<int TNClasses>
class particleFilterPosClassPhaseOri : public particleFilterBase < stateModelOriAwarePosClass<TNClasses,2> ,
																   stateModelPhase<TNClasses,0> ,
																   stateModelOri<TNClasses,0>  >
{
	public:
		// Construtors
		particleFilterPosClassPhaseOri();
		particleFilterPosClassPhaseOri(const int ysize, const int xsize, size_t n_particles,
									const double radius, const double frame_rate,
									const std::string& def_file_posclass, const std::string& def_file_phase, cv::Mat_<unsigned char>* const valid_mask);

		void visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const;

};

#include "particleFilterPosClassPhaseOri.tpp"

#endif
