#ifndef MP_TRANSFORM_HPP
#define MP_TRANSFORM_HPP

// Define a transform function that can transform the type of each element of a
// tuple according to some operation (e.g. selecting a sub-type)

namespace thesisUtilities
{

// Declaration
template< template<class...> class TTransform, class TInitial >
struct mp_transform_impl;

// Use partial specialisation
template<template<class...> class TTransform, template<class...> class TContainer, class... T>
struct mp_transform_impl<TTransform, TContainer<T...>>
{
	// Use a slightly unconventional parameter pack expansion
	using type = TContainer<TTransform<T>...>;
};

// Alias this
template<template<class...> class TTransform, class TInitial>
	using mp_transform = typename mp_transform_impl<TTransform, TInitial>::type;

} // end namespace

#endif
