#include "windows.h"	// for gdiplus to compile

#include <time.h>

#include "IntegralImage.h"

void GenerateRandRect(const int WIDTH, const int HEIGHT
					  ,int& left, int& right, int& top, int& bottom)
{
	left = rand() % WIDTH;	top = rand() % HEIGHT;	
	right = rand() % (WIDTH+1);	bottom = rand() % (HEIGHT+1);
	if (left == right) 
	{
		left = 0;
		right = WIDTH;
	}
	else if (left > right) swap(left, right);

	if (top == bottom)
	{
		top = 0;
		bottom = HEIGHT;
	}
	else if (top > bottom) swap(top, bottom);
}

class IntegralImage_InputImage_Concept
{
public:
	typedef int PixelType;

	unsigned int Width() const { return 0; }
	unsigned int Height() const { return 0; }
	const PixelType* RowPtr(int y) const
	{
		return static_cast<const PixelType*>(0);
	}
};

void test_integral_image_concept()
{
	IntegralImage_InputImage_Concept input_image;

	IntegralImage<IntegralImage_InputImage_Concept::PixelType> integral_image;

	integral_image.Create(input_image);
}

//#include <img\Img_Gdiplus.h>
//using namespace img;
//
//void test_integral_image()
//{	
//	CImage<unsigned int> image;
//	//*
//	const int WIDTH = 3;
//	const int HEIGHT = 3;
//	image.Allocate(3,3);
//	image.Pixel(0, 0) = 1;	image.Pixel(1, 0) = 2;	image.Pixel(2, 0) = 3;
//	image.Pixel(0, 1) = 4;	image.Pixel(1, 1) = 5;	image.Pixel(2, 1) = 6;
//	image.Pixel(0, 2) = 7;	image.Pixel(1, 2) = 8;	image.Pixel(2, 2) = 9;
//
//	// correct integral image should be [1 3 6; 5 12 21; 12 27 45]
//	/*/
//	const int WIDTH = 9;
//	const int HEIGHT = 5;
//	time_t t;	srand(time(&t));
//	int temp = 0;
//	image.Allocate(WIDTH, HEIGHT);
//	for(int y = 0; y < HEIGHT; y++)
//		for(int x = 0; x < WIDTH; x++)
//			image.Pixel(x, y) = temp++;	//rand() % 256;
//	//*/
//
//	IntegralImageUInt int_image;
//	int_image.Create(image);
//
//	const int N_TRIAL = 100;
//	bool success;
//
//	cout << "test intensity integral image for " << N_TRIAL << " trials" << endl;
//	success = true;
//	for(int nTrial = 0; nTrial < N_TRIAL; nTrial++)
//	{
//		int left, right, top, bottom;
//		GenerateRandRect(WIDTH, HEIGHT, left, right, top, bottom);
//
//		int sum = 0;
//		for(int y = top; y < bottom; y++)
//			for(int x = left; x < right; x++)
//				sum += image.Pixel(x, y);
//		if (sum != int_image.Sum(left, top, right, bottom))
//		{
//			success = false;
//			cout << "fail trial " << sum << " != " << int_image.Sum(left, top, right, bottom)
//				<< " at " << left << " " << right << " " << top << " " << bottom << endl;
//		}
//	}
//	if (success) 
//		cout << "TestIntegralImage() : successful!" << endl;
//	else cout << "TestIntegralImage() : fail!" << endl;
//
//	/*
//	success = true;
//	cout << "test integral image down sample for " << N_TRIAL << " trials" << endl;
//	for(int nFactor = 2; nFactor <= 2; nFactor++)
//	{
//	cout << "test down sample factor " << nFactor << endl;
//	IntegralImageUInt down_sample_int_image(int_image);
//	down_sample_int_image.DownSample(nFactor);
//	down_sample_int_image.AssertValid();
//
//	for(int nTrial = 0; nTrial < N_TRIAL; nTrial++)
//	{
//	int left, right, top, bottom;
//	GenerateRandRect(down_sample_int_image.Width()-2, down_sample_int_image.Height()-2
//	,left, right, top, bottom);
//	int sum1 = down_sample_int_image.Sum(left, top, right, bottom);
//	//
//	int sum2 = int_image.Sum(left*nFactor, top*nFactor, right*nFactor, bottom*nFactor);
//	//
//	if (sum1 != sum2)
//	{
//	success = false;
//	cout << "fail trial " << sum1 << " != " << sum2
//	<< " at " << left << " " << right << " " << top << " " << bottom << endl;
//	}
//	}
//	}
//	if (success) cout << "successful!" << endl;
//	*/
//}