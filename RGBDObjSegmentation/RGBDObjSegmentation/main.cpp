//////////////////////////////////////////////////////////////////////////



#include "VideoObjSegmentor.h"


int main()
{
	rgbdvision::VideoObjSegmentor vobj_segmentor;

	string lab_dir = "F:\\test\\fire_processed_new\\";
	string video_dir = "C:\\Users\\vv\\Dropbox\\Experiment data\\book_processed\\book_processed\\";	//"C:\\Users\\jiefeng\\Documents\\MATLAB\\";
	vobj_segmentor.DoRGBDOverSegmentation(lab_dir, 98, 102);
	//vobj_segmentor.DoSegmentation(lab_dir, 98, 102, rgbdvision::SEG_RGB);

	return 0;
}