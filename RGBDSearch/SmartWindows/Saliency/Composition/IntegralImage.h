#pragma once

#include <assert.h>
#include <iostream>
#include <vector>
using namespace std;

template<class T>
struct IntegralImageTypeTraits
{
	typedef T ValueType;
	typedef T AccumulationType;	
};

// specialized for unsigned short for memory efficiency, e.g., for histogram bin counting
// note that it is only valid for small images ( w*h < size(unsigned short) )
template<>
struct IntegralImageTypeTraits<unsigned short>
{
	typedef unsigned short ValueType;
	typedef unsigned int AccumulationType;
};

template<class T> class IntegralImage;
typedef IntegralImage<unsigned int> IntegralImageUInt;
typedef IntegralImage<unsigned short> IntegralImageUShort;
typedef IntegralImage<float> IntegralImageFloat;
typedef IntegralImage<double> IntegralImageDouble;

#include "functional_ex.h"

template<class T>
class IntegralImageInputDummyImage
{
public:
	typedef T PixelType;
	int Width() const { return 0; }
	int Height() const { return 0; }
	const PixelType* RowPtr(int row) const { return NULL; }
};

/*
Usage of IntegralImage
1. Define an image class InputImage that implements the interface as in IntegralImageInputDummyImage.
2. Call IntegralImage::Create() to create an instance of an IntegralImage.
3. Call IntegralImage::Sum() to compute the summation over arbitrary rectangle.

For example, following code compiles successfully (but causes runtime error). 
You should use your own implementation InputImage to replace IntegralImageInputDummyImage.

IntegralImageFloat integralImage;
IntegralImageInputDummyImage<float> inputImage;
integralImage.Create(inputImage);
int left, top, right, bottom;
integralImage.Sum(left, top, right, bottom);
*/

template<class T>
class IntegralImage
{
	// integral image interface
public:
	typedef typename IntegralImageTypeTraits<T>::ValueType ValueType;
	typedef typename IntegralImageTypeTraits<T>::AccumulationType AccumulationType;

	// return sum at rect[(left, top), (right, bottom))
	AccumulationType Sum(int left, int top, int right, int bottom) const
	{
		assert("IntegralImage::Sum()" && (right >= left));
		assert("IntegralImage::Sum()" && (bottom >= top));
		return At(bottom, right) - At(bottom, left) - At(top, right) + At(top, left);
	}
	// wrapper for rect
	template<class Rectangle>
	AccumulationType Sum(const Rectangle& rect) const	
	{	return Sum(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);	}

	template<class PixelFunc, class ImageType>
	void Compute(const ImageType& img, const PixelFunc& fun)
	{
		const int WIDTH = img.Width();
		const int HEIGHT = img.Height();
		Allocate(WIDTH+1, HEIGHT+1);

		/*
		// 1. naive horizontal scan accumulation
		for (int y = 0; y < HEIGHT; y++)
		{
			const ImageType::PixelType* pRow = img.RowPtr(y);
			for(int x = 0; x < WIDTH; x++, pRow++)
				At(y+1, x+1) = At(y+1, x) + At(y, x+1) - At(y, x) + fun(*pRow);
		}
		/*/
		// 2. 3 times faster by storing a row sum
		ValueType row_sum = ValueType(0);
		// set first row
		ValueType* pIImgRow = &At(1, 1);
		const ImageType::PixelType* pRow = img.RowPtr(0);
		for (int x = 0; x < WIDTH; x++, pIImgRow++, pRow++) 
			*pIImgRow = (row_sum += fun(*pRow));

		for (int y = 1; y < HEIGHT; y++)
		{
			row_sum = 0;
			ValueType* pIImgLastRow = &At(y, 1);
			pIImgRow = &At(y+1, 1);
			pRow = img.RowPtr(y);
			for (int x = 0; x < WIDTH; x++, pIImgRow++, pIImgLastRow++, pRow++)
				*pIImgRow = (row_sum += fun(*pRow)) + (*pIImgLastRow);
		}
		//*/
	}

	// by default, use Create() to create the integral image
	template<class ImageType>
	void Create(const ImageType& img)
	{
		Compute(img, unary_identity<ImageType::PixelType>());
	}	

	// create an integral image corresponding to a bin in an integral histogram
	template<class ImageType>
	void CreateBin(const ImageType& bin_img, unsigned int bin)
	{
		Compute(bin_img, unary_equal_to<unsigned int>(bin));
	}	
	
	// auxiliary functions
public:	
	unsigned int Height() const { return static_cast<unsigned int>(m_data.size()); }
	unsigned int Width() const { return m_data.empty() ? 0 : static_cast<unsigned int>(m_data[0].size()); }
	float MemoryInKB() const { return Width() * Height() * sizeof(ValueType) / 1024.0f; }
	bool AssertValid() const
	{
		for(unsigned int y = 0; y < Height()-1; y++)
			for(unsigned int x = 0; x < Width()-1; x++)
			{	
				assert(At(y,x) <= At(y,x+1));
				assert(At(y,x) <= At(y+1,x));
				if (At(y,x) > At(y,x+1))
				{
					cout << "invalid integral image value : " << At(y,x) << " at " << x << " " << y
						<< " > " << At(y,x+1) << " at " << x+1 << " " << y << endl;
					return false;
				}
				if (At(y,x) > At(y+1,x))
				{
					cout << "invalid integral image value : " << At(y,x) << " at " << x << " " << y
						<< " > " << At(y+1,x) << " at " << x << " " << y+1 << endl;
					return false;
				}
			}
			return true;
	}

	// 2D array implementation, could be decoupled from the integral image interface
private:
	// element At(y, x) stores the sum accumulated over rect[(0,0),(x-1,y-1)]
	const ValueType& At(int y, int x) const	{	return m_data[y][x];	}
	ValueType& At(int y, int x)	{	return m_data[y][x];	}
	vector< vector< ValueType > > m_data;
	void Allocate(int width, int height)	{	m_data.resize(height, vector<ValueType>(width, ValueType(0)));	}
};

//void test_integral_image();