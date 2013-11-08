//////////////////////////////////////////////////////////////////////////



#include "VideoObjSegmentor.h"


int main()
{
	rgbdvision::VideoObjSegmentor vobj_segmentor;

	string lab_dir = "C:\\Users\\jiefeng\\Dropbox\\Experiment data\\book_processed\\book_processed\\";
	string video_dir = "C:\\Users\\vv\\Dropbox\\Experiment data\\book_processed\\book_processed\\";	//"C:\\Users\\jiefeng\\Documents\\MATLAB\\";

	vobj_segmentor.DoSegmentation(video_dir, 0, 26);

	return 0;
}