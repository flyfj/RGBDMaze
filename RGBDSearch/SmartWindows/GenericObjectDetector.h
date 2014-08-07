//////////////////////////////////////////////////////////////////////////
// generic object detector
// jiefeng@2014-3-14
//////////////////////////////////////////////////////////////////////////

#pragma once

#include "common.h"
#include "ImgVisualizer.h"
//#include "WindowEvaluator.h"
//#include "ImageSpaceManager.h"
#include "DataManager/DatasetManager.h"
#include "ObjectSegmentor.h"
#include "a9wins/A9Window.h"
#include "WindowEvaluator.h"
#include "Bing/Objectness.h"
#include "Saliency/Composition/SalientRGBDRegionDetector.h"


struct WinConfig
{
	int width;
	int height;
	WinConfig(int w, int h) { width = w; height = h; }
};

class GenericObjectDetector
{
private:

	Mat Gx, Gy;
	Mat Gmag, Gdir;
	Mat integralGx, integralGy;
	vector<Mat> colorIntegrals;
	Size imgSize;
	Mat depthMap;
	Mat img;

	cv::TermCriteria shiftCrit;

	DatasetManager db_man;
	visualsearch::ImageSegmentor segmentor;

	A9Window a9win;

	vector<WinConfig> winconfs;	// sliding window configurations

	//////////////////////////////////////////////////////////////////////////

	Objectness* bingObjectness;
	bool isBingInitialized;

	// window shifting tools
	// given a segment box, window size and image size, the range of window locations 
	// are computed to include the segment
	bool WinLocRange(const Rect spbox, const WinConfig winconf, Point& minPt, Point& maxPt);

	bool SampleWinLocs(const Point startPt, const WinConfig winconf, const Point minPt, const Point maxPt, int num, vector<ImgWin>& wins);

	bool ShiftWindow(const Point& seedPt, Size winSz, Point& newPt);

	bool ShiftWindowToMaxScore(const Point& seedPt, Point& newPt);

	double ComputeObjectScore(Rect win);

	double ComputeCenterSurroundMeanColorDiff(ImgWin win);

	double ComputeDepthVariance(Rect win);

public:
	
	GenericObjectDetector(void);

	~GenericObjectDetector();

	//////////////////////////////////////////////////////////////////////////
	// bing related methods
	//////////////////////////////////////////////////////////////////////////

	bool InitBingObjectness();

	bool TrainBing();

	bool GetObjectsFromBing(const cv::Mat& cimg, vector<ImgWin>& detWins, int winnum, bool showres=false);

	//////////////////////////////////////////////////////////////////////////
	// tool functions
	bool CreateScoremapFromWins(int imgw, int imgh, const vector<ImgWin>& imgwins, Mat& scoremap);

	//////////////////////////////////////////////////////////////////////////
	// main processing functions

	bool Preprocess(const cv::Mat& color_img);

	bool test();

	bool RunVOC();

	bool Run(const cv::Mat& color_img, vector<ImgWin>& det_wins);
	
	bool RunSlidingWin(const cv::Mat& color_img, Size winsz);

	//////////////////////////////////////////////////////////////////////////
	// output ranked object candidates
	bool ProposeObjects(const Mat& cimg, const Mat& dmap, vector<ImgWin>& objwins, vector<ImgWin>& salwins, bool ifRank = true);

};

