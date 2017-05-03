#ifndef MP_TYPE_REPEATER_HPP
#define MP_TYPE_REPEATER_HPP

// Creating a fully generic type repeater
// Copied (mostly) from http://www.preney.ca/paul/archives/279
// there is a good explanation there

// To use this to create, e.g. a std::tuple of four int called my_tup, do this
// mp_type_repeater<4,std::tuple,int> my_tup;

namespace thesisUtilities
{

// ---------------------------------------
// Helper struct (used by the implementation)
template
<
	unsigned N,
	template <typename...> class T,
	typename Arg,
	typename... Args
>
struct mp_type_repeater_helper
{
	// This class was invoked with Arg, Args... and to increase the
	// Arg one more time we need to add it again which is why Arg
	// appears twice below...
	using type = typename mp_type_repeater_helper<N-1,T,Arg,Arg,Args...>::type;
};

template
<
	template <typename...> class T,
	typename Arg,
	typename... Args
>
struct mp_type_repeater_helper<0,T,Arg,Args...>
{
	// Base case: Stop the recursion and expose Arg.
	using type = T<Arg,Args...>;
};

// ---------------------------------------------
// The implementation struct
template <unsigned N, template <typename...> class T, typename Arg>
struct mp_type_repeater_impl
{
	using type = typename mp_type_repeater_helper<N-1,T,Arg>::type;
};

template <template <typename...> class T, typename Arg>
struct mp_type_repeater_impl<0,T,Arg>
{
	using type =  T<>;
};

// --------------------------------------------------------
// Alias this to allow for easy use
template <unsigned N, template <typename...> class T, typename Arg>
using mp_type_repeater = typename mp_type_repeater_impl<N,T,Arg>::type;

} // end namespace

#endif
