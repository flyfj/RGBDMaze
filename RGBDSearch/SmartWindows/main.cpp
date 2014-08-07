

//#include "ImageSpaceManager.h"
//#include "WindowEvaluator.h"
#include "GenericObjectDetector.h"
#include "ShapeAnalyzer.h"
#include "ImgVisualizer.h"
#include "DataManager/DatasetManager.h"
#include "DataManager/NYUDepth2DataMan.h"
#include <string>
#include "ObjectSegmentor.h"
#include "a9wins/A9Window.h"
#include "Saliency/Composition/SalientRegionDetector.h"
#include "Saliency/Composition/SalientRGBDRegionDetector.h"
#include "ObjectTester.h"
#include "Saliency/Depth/DepthSaliency.h"
#include "Tester.h"
using namespace std;

int main()
{
	Tester tester1;
	//tester1.TestFixationSegmentation();
	tester1.TestViewSearch();
	return 0;


	ObjectTester tester;
	//tester.TestObjectRanking(DB_NYU2_RGBD);
	tester.RunVideoDemo();
	return 0;

	ShapeAnalyzer shaper;
	GenericObjectDetector detector;
	DatasetManager dbMan;
	dbMan.Init(DB_VOC07);
	visualsearch::ImageSegmentor segmentor;
	Berkeley3DDataManager b3dman;
	NYUDepth2DataMan nyuman;
	DepthSaliency dsal;
	SalientRGBDRegionDetector saldepth;

	// process
	if( !detector.InitBingObjectness() )
		return -1;

	string b3ddir = "E:\\Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\KinectColor\\";
	string b3dddir = "E:\\Datasets\\RGBD_Dataset\\Berkeley\\VOCB3DO\\RegisteredDepthData\\";
	string datadir = "E:\\Datasets\\RGBD_Dataset\\NYU\\Depth2\\";
	string imgfn = "img_0632.jpg";
	Mat timg = imread(datadir + imgfn);
	if(timg.empty())
		return 0;

	Mat dimg;
	string dmapfn = b3dddir + "img_0632_abs_smooth.png";
	b3dman.LoadDepthData(dmapfn, dimg);
	
	//dimg = dimg * 1000;
	//Mat cloud;
	//dsal.DepthToCloud(dimg, cloud);
	////dsal.OutputToOBJ(cloud, "temp.obj");
	//return 0;

	FileInfos imgfns;
	FileInfo tmpfns;
	tmpfns.filename = imgfn;
	tmpfns.filepath = datadir + imgfn;
	imgfns.push_back(tmpfns);
	map<string, vector<ImgWin>> rawgtwins;
	b3dman.LoadGTWins(imgfns, rawgtwins);
	vector<ImgWin> gtwins = rawgtwins[imgfn];

	//resize(timg, timg, Size(200,200));
	//imshow("input img", timg);
	visualsearch::ImgVisualizer::DrawFloatImg("dmap", dimg, Mat());
	visualsearch::ImgVisualizer::DrawImgWins("gt", timg, gtwins);
	waitKey(10);
	//Mat normimg;
	//normalize(timg, timg, 0, 255, NORM_MINMAX);

	//////////////////////////////////////////////////////////////////////////
	// get objectness proposal

	double start_t = cv::getTickCount();

	vector<ImgWin> boxes;

	detector.GetObjectsFromBing(timg, boxes, 500);

	Mat objectnessmap;
	//detector.CreateScoremapFromWins(timg.cols, timg.rows, boxes, objectnessmap);
	//visualsearch::ImgVisualizer::DrawFloatImg("objmap", objectnessmap, objectnessmap);

	std::cout<<"Bing time: "<<(cv::getTickCount()-start_t) / cv::getTickFrequency()<<"s."<<std::endl;
	
	// make images
	vector<Mat> imgs(boxes.size());
	for (int i=0; i<boxes.size(); i++)
	{
		imgs[i] = timg(boxes[i]);
	}

	Mat dispimg;
	visualsearch::ImgVisualizer::DrawImgCollection("objectness", imgs, 50, 15, dispimg);
	imshow("objectness", dispimg);
	visualsearch::ImgVisualizer::DrawImgWins("objdet", dimg, boxes);
	waitKey(10);
	
	// rank windows with CC
	/*SalientRegionDetector salDetector;
	salDetector.Init(timg);

	start_t = getTickCount();

	salDetector.RankWins(boxes);*/
	
	//////////////////////////////////////////////////////////////////////////
	// depth ranking
	normalize(dimg, dimg, 0, 255, NORM_MINMAX);
	if( !saldepth.Init(SAL_COLOR, timg, dimg) )
		cout<<"Error initialize depth saliency."<<endl;
	saldepth.RankWins(boxes);

	Mat salmap;
	//detector.CreateScoremapFromWins(timg.cols, timg.rows, boxes, salmap);
	//visualsearch::ImgVisualizer::DrawFloatImg("salmap", salmap, salmap);

	std::cout<<"Saliency time: "<<(cv::getTickCount()-start_t) / cv::getTickFrequency()<<"s."<<std::endl;

	// make images
	vector<ImgWin> topBoxes;
	for (int i=0; i<boxes.size(); i++)
	{
		cout<<boxes[i].score<<endl;
		imgs[i] = timg(boxes[i]);
		if(i<15)
			topBoxes.push_back(boxes[i]);
	}

	visualsearch::ImgVisualizer::DrawImgCollection("objectness", imgs, 50, 15, dispimg);
	imshow("rank by CC", dispimg);
	visualsearch::ImgVisualizer::DrawImgWins("ddet", dimg, topBoxes);
	//visualsearch::ImgVisualizer::DrawImgWins("cdet", timg, topBoxes);
	waitKey(0);

	cv::destroyAllWindows();

	//getchar();

	return 0;
}