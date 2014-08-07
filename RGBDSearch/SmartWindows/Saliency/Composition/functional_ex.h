#pragma once

#include <functional>
#include <string>
#include <valarray>
using namespace std;

template<class _Ty>
struct unary_identity : public unary_function<_Ty, _Ty>
{
	_Ty operator()(const _Ty& _Left) const	{	return _Left;	}
};

template<class _Ty>
struct unary_equal_to : public binder2nd<equal_to<_Ty>>
{
	unary_equal_to(const _Ty& _Right) : binder2nd<equal_to<_Ty>>(equal_to<_Ty>(), _Right) {}
};

template<class _Ty>
struct unary_multiply : public binder2nd<multiplies<_Ty>>
{	
	unary_multiply(const _Ty& _Right) : binder2nd<multiplies<_Ty>>(multiplies<_Ty>(), _Right) {}
};

template<class _Ty>
struct unary_entropy : public unary_function<_Ty, _Ty>
{
	_Ty operator()(const _Ty& _Left) const	{	return (_Left > 1e-6) ? (-_Left*log(_Left)) : 0.0f;	}
};

// fast by pre-computation on a fixed domain [0, N]
template<class IntType, class _Ty>
struct unary_entropy_int_fast : public unary_function<IntType, _Ty>
{
protected:
	valarray<float> entropy_result_table;
	void pre_compute()
	{
		entropy_result_table[0] = 0.0f;
		for(int n = 1; n < entropy_result_table.size(); n++)
			entropy_result_table[n] = n * logf(n);
	}

public:
	unary_entropy_int_fast(IntType N) : entropy_result_table(N+1) {	pre_compute();	}
	_Ty operator()(IntType n) const
	{	
		assert(n >= 0);
		assert(n < entropy_result_table.size());
		return entropy_result_table[n];
	}
};

inline float fast_sqrt(const float& arg)
{
	float		x, y;
	unsigned	iarg = *((unsigned*)&arg);
	unsigned	SQRT_MAGIC = 0xbe6f0000;

	*((unsigned*)&x) = (SQRT_MAGIC - iarg)>>1;
	y = arg*0.5f;
	x*= 1.5f - y*x*x;
	x*= 1.5f - y*x*x;

	return x*arg;
}

template<class _Ty>
struct binary_null : public binary_function<_Ty, _Ty, _Ty>
{
	enum { IS_0_WHEN_OP1_IS_0 = true };
	enum { IS_0_WHEN_OP2_IS_0 = true };
	static string Name() {	return "Null";	}
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	{	return (_Ty)0;	}
};

template<class _Ty>
struct binary_min : public binary_function<_Ty, _Ty, _Ty>
{
	static string Name() {	return "Inter";	}
	enum { IS_0_WHEN_OP1_IS_0 = true };
	enum { IS_0_WHEN_OP2_IS_0 = true };
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	{	return (_Left < _Right) ? _Left : _Right;	}
};

template<class _Ty>
struct binary_abs : public binary_function<_Ty, _Ty, _Ty>
{
	static string Name() {	return "L1Norm";	}
	enum { IS_0_WHEN_OP1_IS_0 = false };
	enum { IS_0_WHEN_OP2_IS_0 = false };
	// fabs() is much faster than : v >= 0 ? v : -v;
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	{	return fabs(_Left - _Right);	}
};

template<class _Ty>
struct binary_sqr_abs : public binary_function<_Ty, _Ty, _Ty>
{
	static string Name() {	return "L2Norm";	}
	enum { IS_0_WHEN_OP1_IS_0 = false };
	enum { IS_0_WHEN_OP2_IS_0 = false };
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	{	_Ty d = _Left - _Right;	return d*d;	}
};

template<class _Ty>
struct binary_chi_sqr : public binary_function<_Ty, _Ty, _Ty>
{
	static string Name() {	return "ChiSquare";	}
	enum { IS_0_WHEN_OP1_IS_0 = false };
	enum { IS_0_WHEN_OP2_IS_0 = false };
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	
	{	
		//if (_Left == 0) return _Right;	// useful when _Left is likely to be 0, e.g., sparse histogram bins
		//if (_Right == 0) return _Left;	// useful when _Right is likely to be 0
		_Ty denominator = _Left + _Right;
		return (denominator < 1e-6) ? 0 : ((_Left-_Right) * (_Left-_Right) / denominator);
	}
};

template<class _Ty>
struct binary_dot : public multiplies<_Ty>
{
	static string Name() {	return "Dot";	}
	enum { IS_0_WHEN_OP1_IS_0 = true };
	enum { IS_0_WHEN_OP2_IS_0 = true };
};

template<class _Ty>
struct binary_sqrt_multiply : public binary_function<_Ty, _Ty, _Ty>
{
	static string Name() {	return "BHSim";	}
	enum { IS_0_WHEN_OP1_IS_0 = true };
	enum { IS_0_WHEN_OP2_IS_0 = true };
	// sqrt() is even faster than fast_sqrt
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	{	return sqrt(_Left * _Right);	}
};

template<class _Ty>
struct binary_max_of_diff_and_zero : public binary_function<_Ty, _Ty, _Ty>
{
	static string Name() {	return "MaxDiffZero";	}

	// the constants depend on whether _Ty is signed, here we assume it is unsigned
	enum { IS_0_WHEN_OP1_IS_0 = true };
	enum { IS_0_WHEN_OP2_IS_0 = false };

	// should not use (_Left -_Right <= 0) as _Ty could be unsigned
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const	{	return (_Left <= _Right) ? 0 : (_Left - _Right);	}
};

// basic bin-2-bin operator to compute similarity between histograms
enum Bin2BinOperator
{
	BIN_OP_NULL				=	1	
	,BIN_OP_INTERSECTION
	// dot is not suitable for similarity calculation due to inappropriate normalization, use BhattachSimilarity instead
	// however, it could be used for linear SVM
	,BIN_OP_DOT				
	,BIN_OP_BH
	,BIN_OP_L1NORM
	,BIN_OP_L2NORM
	,BIN_OP_CHISQUARE
};

inline string BinOp2Name(int bin_op)
{
	switch (bin_op)
	{
	case BIN_OP_NULL:	return binary_null<int>::Name();
	case BIN_OP_INTERSECTION: return binary_min<int>::Name();
	case BIN_OP_DOT:	return binary_dot<int>::Name();
	case BIN_OP_BH:		return binary_sqrt_multiply<int>::Name();
	case BIN_OP_L1NORM: return binary_abs<int>::Name();	
	case BIN_OP_L2NORM: return binary_sqr_abs<int>::Name();
	case BIN_OP_CHISQUARE: return binary_chi_sqr<int>::Name();
	default:	return "UnkownBinOP";
	}
}