#ifndef MP_TUPLE_REPEAT_H
#define MP_TUPLE_REPEAT_H

// Creates a runtime tuple object by repeating a value a fixed number of
// times. To use do e.g.
//
// int x = 3;
// auto tup = tuple_repeat<4>(3);
//
// tup is now a std::tuple<int,int,int,int> containing [3,3,3,3]

namespace thesisUtilities
{

template<std::size_t> struct int2type{};

template <size_t I, class T>
auto tuple_repeat_impl(int2type<I>, T t)
{
	return std::tuple_cat(std::make_tuple(t), tuple_repeat_impl(int2type<I-1>(), t));
}

template <class T>
auto tuple_repeat_impl(int2type<0>, T t)
{
	return std::make_tuple(t);
}

template <unsigned N, class T>
auto mp_tuple_repeat(T t)
{
	return tuple_repeat_impl(int2type<N-1>(),t);
}

} // end namespace

#endif
