#ifndef PARTICLEFILTERJOINEDSTRUCTSPCA_HPP
#define PARTICLEFILTERJOINEDSTRUCTSPCA_HPP

#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelPosClassOri.hpp"
#include "stateModelPhase.hpp"
#include "stateModelSubstructuresPCA.hpp"



template<int TNClasses>
class particleFilterJoinedStructsPCA : public particleFilterBase  <
															  stateModelPosClassOri<TNClasses> ,
															  stateModelPhase<TNClasses,0> ,
															  stateModelSubstructuresPCA<TNClasses,0,1,0>
															>
{
	public:
		// Construtors
		particleFilterJoinedStructsPCA();
		particleFilterJoinedStructsPCA(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
									const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs,
									const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask);

		void visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const;
		void visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const;

};

#include "particleFilterJoinedStructsPCA.tpp"

#endif
