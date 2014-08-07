#pragma once

#include <vector>
using namespace std;

// a simple templated image interface, mainly for convenience
// should be easily replaced by other implementation for performance reasons
template<class P> class ImageSimple;

typedef ImageSimple<float> ImageFloatSimple;
typedef ImageSimple<unsigned int> ImageUIntSimple;

template<class P>
class ImageSimple
{
public:
	typedef P PixelType;
	
	ImageSimple() : width(0) {}
	ImageSimple(unsigned int w, unsigned int h) : data(h*w, 0), width(w)
	{
		assign_row_ptrs(h);
	}

	void Create(unsigned int w, unsigned int h)	
	{			
		width = w;
		data.resize(h*w, 0);
		assign_row_ptrs(h);
	}

	void FillPixels(const PixelType& v)
	{
		for(int i = 0; i < data.size(); i++)
			data[i] = v;
	}

	unsigned int Height() const { return row_ptrs.size(); }
	unsigned int Width() const { return width; }

	const PixelType* RowPtr(int y) const	{ return row_ptrs[y]; }
	const PixelType& Pixel(int x, int y) const {	return row_ptrs[y][x];	}
	PixelType& Pixel(int x, int y) {	return row_ptrs[y][x];	}

private:
	// inhibit copy constructor due to pointer members
	ImageSimple(const ImageSimple& im) {}
	ImageSimple& operator = (const ImageSimple& im) { return *this; }

	vector<PixelType> data;
	vector<PixelType*> row_ptrs;
	int width;

	void assign_row_ptrs(int h)
	{
		row_ptrs.resize(h);

		for(int i = 0; i < h; i++)
			row_ptrs[i] = &data[i*width];
	}
};