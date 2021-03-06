#pragma once

class FilterTIG
{
public:
	void update(CMat &w);	

	// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
	Mat matchTemplate(const Mat &mag1u); 
	
	inline float dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8);

public:
	void reconstruct(Mat &w); // For illustration purpose

private:
	static const int NUM_COMP = 2; // Number of components
	static const int D = 64; // Dimension of TIG
	INT64 _bTIGs[NUM_COMP]; // Binary TIG features
	float _coeffs1[NUM_COMP]; // Coefficients of binary TIG features

	// For efficiently deals with different bits in CV_8U gradient map
	float _coeffs2[NUM_COMP], _coeffs4[NUM_COMP], _coeffs8[NUM_COMP]; 
};


inline unsigned __int64 __popcount64(unsigned __int64 x)
{
	return
		__popcnt((unsigned int)(x )) +
		__popcnt((unsigned int)(x >> 32));
}

#define USE_32

inline float FilterTIG::dot(const INT64 tig1, const INT64 tig2, const INT64 tig4, const INT64 tig8)
{
#ifdef USE_64
	INT64 bcT1 = __popcnt64(tig1);
	INT64 bcT2 = __popcnt64(tig2);
	INT64 bcT4 = __popcnt64(tig4);
	INT64 bcT8 = __popcnt64(tig8);

	INT64 bc01 = (__popcnt64(_bTIGs[0] & tig1) << 1) - bcT1;
	INT64 bc02 = ((__popcnt64(_bTIGs[0] & tig2) << 1) - bcT2) << 1;
	INT64 bc04 = ((__popcnt64(_bTIGs[0] & tig4) << 1) - bcT4) << 2;
	INT64 bc08 = ((__popcnt64(_bTIGs[0] & tig8) << 1) - bcT8) << 3;

	INT64 bc11 = (__popcnt64(_bTIGs[1] & tig1) << 1) - bcT1;
	INT64 bc12 = ((__popcnt64(_bTIGs[1] & tig2) << 1) - bcT2) << 1;
	INT64 bc14 = ((__popcnt64(_bTIGs[1] & tig4) << 1) - bcT4) << 2;
	INT64 bc18 = ((__popcnt64(_bTIGs[1] & tig8) << 1) - bcT8) << 3;
#endif

#ifdef USE_32
	INT64 bcT1 = __popcount64(tig1);
	INT64 bcT2 = __popcount64(tig2);
	INT64 bcT4 = __popcount64(tig4);
	INT64 bcT8 = __popcount64(tig8);

	INT64 bc01 = (__popcount64(_bTIGs[0] & tig1) << 1) - bcT1;
	INT64 bc02 = ((__popcount64(_bTIGs[0] & tig2) << 1) - bcT2) << 1;
	INT64 bc04 = ((__popcount64(_bTIGs[0] & tig4) << 1) - bcT4) << 2;
	INT64 bc08 = ((__popcount64(_bTIGs[0] & tig8) << 1) - bcT8) << 3;

	INT64 bc11 = (__popcount64(_bTIGs[1] & tig1) << 1) - bcT1;
	INT64 bc12 = ((__popcount64(_bTIGs[1] & tig2) << 1) - bcT2) << 1;
	INT64 bc14 = ((__popcount64(_bTIGs[1] & tig4) << 1) - bcT4) << 2;
	INT64 bc18 = ((__popcount64(_bTIGs[1] & tig8) << 1) - bcT8) << 3;
#endif

	return _coeffs1[0] * (bc01 + bc02 + bc04 + bc08) + _coeffs1[1] * (bc11 + bc12 + bc14 + bc18);
}
