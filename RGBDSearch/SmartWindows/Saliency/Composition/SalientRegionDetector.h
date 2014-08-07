#pragma once

#include "SegmentBasedWindowComposer.h"
#include <opencv2/opencv.hpp>
#include <tchar.h>
#include "ImageSegmentor.h"
#include "colorconverthelper.h"
using namespace cv;


struct DetectionParams
{	
	string g_imagefile, g_outfile;	// input dir and output dir
	double downSampleFactor;	// image downsample para, < 1: in percentage, > 1: in pixel
	
	// segmentation
	double segSigma;	// for smooth
	double segThresholdK;	// segmentation threshold for graph-based method
	double segMinArea;	//	minimum segment area

	int slidingStep;	// 0: use incremental update; 1: use integral image; >1: brute force

	float nmsTh;		// used in non maximum suppression
	int useMultiNMS;		// sign: use multiple nms values
	float nms_min;	// nms lower bound
	float nms_max;	// nms upper bound
	float nms_step;	// nms step

	float saliencyThre;		// salient object with score larger than it is written to file
	int bestN_drawn;		// bestN windows that are drawn onto the image

	int useBGMap;	// whether save background probability map
	int saveSegmap;	// save segmentation image

	// init
	DetectionParams() : downSampleFactor(300), useBGMap(0), saveSegmap(0),  
		segSigma(0.5f), segThresholdK(200), segMinArea(100), slidingStep(1),
		useMultiNMS(0), nmsTh(0.6f), nms_min(0.6f), nms_max(0.6f), nms_step(1), 
		saliencyThre(0.08f), bestN_drawn(5) {}
};

wstring string2wstring(string str);

inline float generate_equiratio_array(float min_value, float max_value, unsigned int levels, vector<float>& a)
{
	assert(levels >= 2);
	float ratio = pow(max_value/min_value, 1.0f/(levels-1));	
	a.resize(levels);

	a[0] = min_value;
	for(unsigned int n = 1; n < levels; n++)
		a[n] = a[n-1] * ratio;

	return ratio;
}

struct SlideWinPara
{		
	vector<float> areas, aspect_ratios;
	float area_ratio, aspect_ratio_ratio;

	SlideWinPara() : area_ratio(1.0f), aspect_ratio_ratio(1.0f)
	{
		areas.push_back(0.1f);
		aspect_ratios.push_back(1.0f);
	}

	SlideWinPara(float min_area, float max_area, int area_step, float min_asp, float max_asp, int asp_step)
	{
		generate_equiratio_array(min_area, max_area, area_step, areas);
		generate_equiratio_array(min_asp, max_asp, asp_step, aspect_ratios);
	}

	float SetArea(float min, float max, int step)
	{
		return area_ratio = generate_equiratio_array(min, max, step, areas);
	}

	float SetAspectRatio(float min, float max, int step)
	{
		return aspect_ratio_ratio = generate_equiratio_array(min, max, step, aspect_ratios);
	}
};

// save necessary information while running
struct RuntimeInfo
{
	// time data
	float downsize_t;	// downsize time
	float init_t;	// init time
	float seg_t;	// segmentation time
	int seg_num;	// number
	float det_t;	// detection time (sliding window)
	int det_num;	// detected results number
	float tol_img_t;	// total run time

	RuntimeInfo(): downsize_t(0), init_t(0), seg_t(0), seg_num(0),
		det_t(0), det_num(0), tol_img_t(0) {}
};

// patch structure (bg map)
struct Patch
{
	int id;	// start from 0
	Rect box;
	float bgScore;	//background score
	Point pos;	// patch coordinates (x,y)
};

// it holds the members necessary to extract salient windows from composition score maps
class SalientRegionDetector : public SegmentBasedWindowComposer
{
	typedef SegmentBasedWindowComposer Base;

public:
	SalientRegionDetector();
	~SalientRegionDetector();

	// initialization:
	// 1. compute static segment features
	// 2. compute background weight map
	bool Init(const Mat& img);	

	void RunSlidingWindow(const int win_width, const int win_height);

	int RunMultiSlidingWindow();	// return how many window scales

	//////////////////////////////////////////////////////////////////////////
	// added@2014-6-6
	//////////////////////////////////////////////////////////////////////////
	// use composition cost to rank image windows
	bool RankWins(vector<ImgWin>& wins);


	// draw detected boxes on image
	void DrawResult(Mat& img, double down_ratio, const vector<ScoredRect>& objs) const;

	int m_nSlideStep;	//?
	
	//bool SaveSegmentImage(const WCHAR* filename) const;

	vector<ScoredRect> res_objs;	// final detection results
	// a number of nms thresholds, and extracted salient objects for each threshold
	vector<float> nms_vals;
	vector<vector<ScoredRect>> all_objs;

	// background weight map
	ImageFloatSimple bgMap;
	
	//////////////////////////////////////////////////////////////////////////
	// cmd params and functions
	DetectionParams g_para;
	SlideWinPara g_win_para;
	RuntimeInfo g_runinfo;

	void print_help();
	bool read_para(int argc, _TCHAR* argv[]);

	// save detection results (windows) to data file
	void SaveDetectionResults(string save_img_prefix);
	// save background image
	void SaveBGMap(string save_img_prefix);

private:
	void Clear();

	void ConvertSegmentImage2Mat(Mat &segmentMat, int width, int height);

	//void ComputeBGMap(const BitmapData& img);

	visualsearch::ImageSegmentor imgSegmentor;

	ImageUIntSimple seg_index_map;

	//	Bitmap color segmentedImg: for visualization
	vector<unsigned char> segmentedImg;	// 3*width*height bgr image buffer

	vector<SegSuperPixelFeature> sp_features;

};