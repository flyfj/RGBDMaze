//////////////////////////////////////////////////////////////////////////



#include "VideoObjSegmentor.h"


int main()
{
	rgbdvision::VideoObjSegmentor vobj_segmentor;

	string lab_dir = "F:\\test\\fire_processed_new\\";
	string video_dir = "C:\\Users\\vv\\Dropbox\\Experiment data\\book_processed\\book_processed\\";	//"C:\\Users\\jiefeng\\Documents\\MATLAB\\";

	vobj_segmentor.DoSegmentation(lab_dir, 73, 73, rgbdvision::SEG_RGBD);

	return 0;
}