#include "wrapper.h"
#include "segment-image.h"


int graph_based_segment(const Mat& img, float sigma, float c, int min_size, Mat& indexImg, Mat& segmentedImg)
{

	int height = img.rows;
	int width = img.cols;
	image<rgb> input(width, height);
	rgb val;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			val.b = img.at<Vec3b>(y, x).val[0];
			val.g = img.at<Vec3b>(y, x).val[1];
			val.r = img.at<Vec3b>(y, x).val[2];
			input.access[y][x] = val;
		}
	}

	image<int> index(width, height);	//index matrix, each pixel value is its object id (0~object_num)
	int num_ccs;
	image<rgb> *seg = segment_image(&input, sigma, c, min_size, &num_ccs, &index);	


	segmentedImg.create(img.size(), CV_8UC3);
	for(int y = 0; y < height; y++)	for(int x = 0; x < width; x++)
	{
		int loc = y*width*3 + 3*x;
		segmentedImg.at<Vec3b>(y,x).val[0] = imRef(seg,x,y).b;
		segmentedImg.at<Vec3b>(y,x).val[1] = imRef(seg,x,y).g;
		segmentedImg.at<Vec3b>(y,x).val[2] = imRef(seg,x,y).r;
	}

	delete seg;


	indexImg.create(height, width, CV_32S);
	for(int y = 0; y < height; y++) for(int x = 0; x < width; x++)
		indexImg.at<int>(y,x) = index.access[y][x];


	return num_ccs;

}

/*
int graph_based_segment(const CImageRgb& img, float sigma, float c, int min_size, CImage<unsigned int>& indexImg, CImageRgb& segmentedImg)
{	
	//copy data to input
	const int width = img.Width();
	const int height = img.Height();
	image<rgb> input(width, height);
	for(int y = 0; y < height; y++) for(int x = 0; x < width; x++)
	{
		PixelRgb t = img.Pixel(x,y);
		rgb val;
		val.r = t.R();
		val.b = t.B();
		val.g = t.G();
		input.access[y][x] = val;
	}
	
	image<int> index(width, height);	//index matrix, each pixel value is its object id (0~object_num)
	int num_ccs;
	image<rgb> *seg = segment_image(&input, sigma, c, min_size, &num_ccs, &index);	
	
	segmentedImg.Allocate(width, height);
	for(int y = 0; y < height; y++)	for(int x = 0; x < width; x++)
		segmentedImg.Pixel(x,y) = PixelRgb(imRef(seg,x,y).r, imRef(seg,x,y).g, imRef(seg,x,y).b);

	delete seg;	
	
	indexImg.Allocate(width, height);
	for(int y = 0; y < height; y++) for(int x = 0; x < width; x++)
		indexImg.Pixel(x,y) = index.access[y][x];

	return num_ccs;
}
*/