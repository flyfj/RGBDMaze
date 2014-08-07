#include "Tester.h"


Tester::Tester(void)
{
}


void Tester::TestFixationSegmentation()
{
	visualsearch::FixationSegmentor fixsegmentor;
	string dmapfn = "E:\\Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\RegisteredDepthData\\img_0016_abs_smooth.png";
	string cimgfn = "E:\\Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\KinectColor\\img_0016.png";
	string tmpfn = "E:\\Datasets\\mail.jpg";
	cout<<tmpfn<<endl;
	Mat cimg = imread(tmpfn);
	Mat dmap = imread(dmapfn);
	resize(cimg, cimg, Size(300, 300));
	resize(dmap, dmap, Size(300, 300));
	fixsegmentor.DoSegmentation(Point(150, 150), cimg, dmap, Mat());
}

void Tester::TestViewSearch()
{
	string fn = "E:\\Datasets\\RGBD_Dataset\\UW\\rgbd-dataset\\flashlight\\flashlight_1\\flashlight_1_1_25_depthcrop.png";
	Mat queryimg = imread(fn, CV_LOAD_IMAGE_ANYDEPTH);
	queryimg.convertTo(queryimg, CV_32F);
	visualsearch::ImgVisualizer::DrawFloatImg("query", queryimg, Mat());
	waitKey(0);

	visualsearch::ViewSearcher vsearcher;
	/*vsearcher.LoadCategoryDepthMaps("E:\\Datasets\\RGBD_Dataset\\UW\\rgbd-dataset\\");

	vsearcher.LearnOptimalBinaryCodes(32);
	vsearcher.BuildHashTable();*/
	 
	vsearcher.LoadSearcher("hashtable.txt");

	//Mat queryimg = imread("E:\\Images\\1_26_26745.jpg");
	vsearcher.Search(queryimg, vector<int>());
	waitKey(0);
}
