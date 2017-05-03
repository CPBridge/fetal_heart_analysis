#ifndef MP_CAT_PARAM_PACK_HPP
#define MP_CAT_PARAM_PACK_HPP

// Utility to take two types formed from a variadic template class
// and return the type that concatenates the two sets of input parameters
//
// e.g. mp_cat_param_pack<std::tuple,std::tuple<int,char>,std::tuple<double,float>>
// becomes std::tuple<int,char,double,float>

namespace thesisUtilities
{

template< template<typename...> class TTemplate, typename TA, typename TB >
struct mp_cat_param_pack_impl;

template< template<typename...> class TTemplate, typename... TA, typename... TB  >
struct mp_cat_param_pack_impl< TTemplate, TTemplate< TA ... >, TTemplate< TB ... > >
{
	using type = TTemplate< TA ..., TB ... >;
};

// Alias for easy use
template< template<typename...> class TTemplate, typename TA, typename TB >
using mp_cat_param_pack = typename mp_cat_param_pack_impl<TTemplate,TA,TB>::type;

} // end namespace

#endif
