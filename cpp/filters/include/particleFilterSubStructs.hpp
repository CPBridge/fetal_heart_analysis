#ifndef PARTICLEFILTERSUBSTRUCTS_HPP
#define PARTICLEFILTERSUBSTRUCTS_HPP

#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelOriAwarePosClass.hpp"
#include "stateModelOri.hpp"
#include "stateModelPhase.hpp"
#include "stateModelSubstructuresPCA.hpp"



template<int TNClasses>
class particleFilterSubStructs : public particleFilterBase  <
                                                              stateModelOriAwarePosClass<TNClasses,2> ,
                                                              stateModelPhase<TNClasses,0> ,
                                                              stateModelOri<TNClasses,0> ,
                                                              stateModelSubstructuresPCA<TNClasses,0,1,2>
                                                            >
{
    public:
        // Construtors
        particleFilterSubStructs();
        particleFilterSubStructs(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
                                    const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs,
                                    const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask);

        void visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const;
        void visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const;

		bool structInView(const int s, const int c) const {return std::get<3>(this->state_models).structInView(s,c-1);}

};

#include "particleFilterSubStructs.tpp"

#endif
