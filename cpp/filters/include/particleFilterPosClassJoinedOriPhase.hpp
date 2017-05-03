#ifndef PARTICLEFILTERPOSCLASSJOINEDORIPHASE_HPP
#define PARTICLEFILTERPOSCLASSJOINEDORIPHASE_HPP

#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelPosClassOri.hpp"
#include "stateModelPhase.hpp"


template<int TNClasses>
class particleFilterPosClassJoinedOriPhase : public particleFilterBase < stateModelPosClassOri<TNClasses> ,
																   stateModelPhase<TNClasses,0>  >
{
	public:
		// Construtors
		particleFilterPosClassJoinedOriPhase();
		particleFilterPosClassJoinedOriPhase(const int ysize, const int xsize, size_t n_particles,
									const double radius, const double frame_rate,
									const std::string& def_file_posclass, const std::string& def_file_phase, cv::Mat_<unsigned char>* const valid_mask);

		void visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const;

};

#include "particleFilterPosClassJoinedOriPhase.tpp"

#endif
