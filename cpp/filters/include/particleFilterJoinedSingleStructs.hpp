#ifndef PARTICLEFILTERJOINEDSINGLESTRUCTS_HPP
#define PARTICLEFILTERJOINEDSINGLESTRUCTS_HPP

#include <tuple>
#include <opencv2/core/core.hpp>
#include "particleFilterBase.hpp"
#include "stateModelPosClassOri.hpp"
#include "stateModelPhase.hpp"
#include "stateModelSingleStructure.hpp"
#include "circCircSingleRegressor.hpp"
#include "orientationTestingFunctor.h"
#include "mp_cat_param_pack.hpp"
#include "mp_type_repeater.hpp"

// Need some meta-programming wizardry to deduce the base type to inherit from
// Stack the components of the heart model, and then a set of structure models
template <int TNClasses, int TNStructs>
using particleFilterJoinedSingleStructsBaseClass =
	thesisUtilities::mp_cat_param_pack
	<
		/* The template type */
		particleFilterBase,
		/* First set of parameters - the heart model */
		particleFilterBase
		<
			/* The position model */
			stateModelPosClassOri<TNClasses> ,
			/* The phase model */
			stateModelPhase<TNClasses,0>
		>,
		/* Second set of parameters - the substructure models */
		thesisUtilities::mp_type_repeater
		<
			/* Number of repetitions */
			TNStructs,
			/* Template type */
			particleFilterBase,
			/* Template parameter */
			stateModelSingleStructure<TNClasses,0,1,0>
		>
	>;



template<int TNClasses, int TNStructs>
class particleFilterJoinedSingleStructs : public particleFilterJoinedSingleStructsBaseClass<TNClasses,TNStructs>
{
	public:
		// Forward this from the base class
		using combined_state_type = typename particleFilterJoinedSingleStructsBaseClass<TNClasses,TNStructs>::combined_state_type ;

		// Construtors
		particleFilterJoinedSingleStructs();
		particleFilterJoinedSingleStructs(const int ysize, const int xsize, const size_t n_particles, const double radius, const double frame_rate,
									const std::string& def_file_posclass, const std::string& def_file_phase, const std::string& def_file_substructs, const std::vector<std::string>& subs_names,
									cv::Mat_<unsigned char>* const valid_mask, cv::Mat_<unsigned char>* const subs_valid_mask);

		void visualiseOriPhase(cv::Mat_<cv::Vec3b>* const disp) const;
		void visualiseSubstructures(cv::Mat_<cv::Vec3b>* const disp) const;

		void structPositionArray(const combined_state_type& s, std::array<double,TNStructs> & y_arr, std::array<double,TNStructs> & x_arr, std::array<structVisible_enum,TNStructs> & visible_arr) const;

	protected:
		// Enables compile time recursion using function overloading
		template<std::size_t> struct int2type{};

		// Recursive functions for initialising substructure models
		template <size_t I> bool initialiseStructModel(int2type<I>, const int ysize, const int xsize, const double radius,
														const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask);
		bool initialiseStructModel(int2type<0>, const int ysize, const int xsize, const double radius, const std::string& def_file_substructs, const std::vector<std::string>& subs_names, cv::Mat_<unsigned char>* const subs_valid_mask);

		template <size_t I>  void structPositionArray_impl(int2type<I>, const combined_state_type& s, std::array<double,TNStructs> & y_arr, std::array<double,TNStructs> & x_arr, std::array<structVisible_enum,TNStructs> & visible_arr) const;
		void structPositionArray_impl(int2type<0>, const combined_state_type& s, std::array<double,TNStructs> & y_arr, std::array<double,TNStructs> & x_arr, std::array<structVisible_enum,TNStructs> & visible_arr) const;
};

#include "particleFilterJoinedSingleStructs.tpp"

#endif
