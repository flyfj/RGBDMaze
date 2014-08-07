#pragma once

//#include <windows.h>
//#include <gdiplus.h>
//using namespace Gdiplus;
#include <opencv2/opencv.hpp>
using namespace cv;

//#include <assert.h>

#include <vector>
using namespace std;

#include "ImageSimple.h"
#include "IntegralImage.h"
#include "common.h"


// make every structure clear; pass only necessary information to algorithm
// 1. redefine data structure, only include truly necessary data members, also differentiate those only for algorithm analysis (using RECORD_AUXILIARY_INFO)
// 2. always initialize variables and make them as private as possible
// 3. make meaningful variable names and add comments to variables and function blocks

// if RECORD_AUXILIARY_INFO is defined, those intermediate results in algorithm are recorded for analysis
// should NOT be defined in the final, and the code still runs correctly

//#define RECORD_AUXILIARY_INFO


enum SalientType
{
	SAL_COLOR = 1,
	SAL_DEPTH = 2,
};

inline float l2_dist(float x1, float y1, float x2, float y2)
{
	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

inline float l2_dist(const Point2f& p1, const Point2f& p2)
{
	return l2_dist(p1.x, p1.y, p2.x, p2.y);
}

struct ScoredRect : public Rect
{
	ScoredRect() : score(0){}
	ScoredRect(const Rect& r) : Rect(r), score(0){}
	ScoredRect(const Rect& r, float s) : Rect(r), score(s){}
	float score;

	static bool comp_by_score(const ScoredRect& a, const ScoredRect& b)
	{
		return a.score < b.score;
	}
};

// information of another segment
struct TPair
{
	TPair(): id(-1), appdist(0), spadist(0), saliency(0)	{}
	
	int id;
	float appdist;	// appearance distance
	float spadist;	// spatial distance
	float saliency;	// pair composition cost

	static bool comp_by_saliency(const TPair& a, const TPair& b) 
	{
		return a.saliency < b.saliency;	
	}
};

// superpixel static features (appearance, perimeter...)
struct SegSuperPixelFeature
{
	SegSuperPixelFeature() : bnd_pixels(0), perimeter(0), 
		area(0), id(-1)
	{ box_pos[0].x = 100000; box_pos[0].y = 100000; box_pos[1].x = 0; box_pos[1].y = 0; }

	vector<float> feat;	// color feature vector
	vector<float> dfeat;	// depth feature vector

	Point box_pos[2];	// bounding box: top-left, bottom-right
	Rect box;	// bounding box
	float bnd_pixels;	// boundary pixel numbers
	float perimeter;
	float area;
	Point2f centroid;
	static bool use4Neighbor;
	unsigned int id;

	static float FeatureIntersectionDistance(const SegSuperPixelFeature& a, const SegSuperPixelFeature& b, SalientType stype = SAL_COLOR)
	{
		float dist = 0;
		if(stype == SAL_COLOR)
		{
			for(size_t i = 0; i < a.feat.size(); i++)
				dist += ((a.feat[i] < b.feat[i]) ? a.feat[i] : b.feat[i]);
		}
		else
		{
			for(size_t i = 0; i < a.dfeat.size(); i++)
				dist += ((a.dfeat[i] < b.dfeat[i]) ? a.dfeat[i] : b.dfeat[i]);
		}
		
		return 1 - dist;
	}
};

// superpixel dynamic features only used for composition
struct SegSuperPixelComposeFeature
{
	SegSuperPixelComposeFeature() : composition_cost(0.f)
		,extendedArea(0), leftInnerArea(0), leftOuterArea(0), currentArea(0)
		,dist_to_win(0), importWeight(1), centroid(0,0)
	{}

	//void Init(const ImageFloatSimple& saliencyMatrix);	

	vector<TPair> pairs;	// list for all other superpixels
	
	float extendedArea;	// area after boundary extension
	float leftInnerArea, leftOuterArea;	// inside and outside areas with respect to a window after self-composing
	float composition_cost;	// cost for composing current superpixel
	float currentArea;	// area need to be filled currently
	float importWeight;	// importance weight: may affected by relative position, background likelihood...	
	Point2f centroid;
	Rect box;	// bounding box
	IntegralImageFloat area_integral_image;
	Mat cv_area_integral_image;
	vector<int> mask;	// mask data (same size as box)

#ifdef RECORD_AUXILIARY_INFO
	vector<TPair> composers;	 // segments composing current sp
#endif
	
	// create mask integral
	/*void CreateAreaIntegral(const Mat& seg_index_map, unsigned int id)
	{
		cv_area_integral_image.create(box.height, box.width, CV_32F);
		Mat mask_img(box.height, box.width, CV_32S);
		mask_img.setTo(0);
		mask.resize(box.width*box.height, 0);
		for(int y=box.y; y<box.br().y; y++)
		{
			for(int x=box.x; x<box.br().x; x++)
			{
				if( seg_index_map.at<int>(y,x) == id )
				{
					mask_img.at<int>(y-box.y, x-box.x) = 1;
					mask[(y-box.y)*box.width+(x-box.x)] = 1;
				}
			}
		}

		integral(mask_img, cv_area_integral_image);
	}*/
	void CreateAreaIntegral(const Mat& seg_index_map, unsigned int id)
	{
		cv_area_integral_image.create(box.height, box.width, CV_32F);
		ImageUIntSimple mask_img(box.width, box.height);
		mask_img.FillPixels(0);
		mask.resize(box.width*box.height, 0);
		for(int y=box.y; y<box.br().y; y++)
		{
			for(int x=box.x; x<box.br().x; x++)
			{
				if( seg_index_map.at<int>(y, x) == id )
				{
					mask_img.Pixel(x-box.x, y-box.y) = 1;
					mask[(y-box.y)*box.width+(x-box.x)] = 1;
				}
			}
		}
		area_integral_image.Create(mask_img);
	}

	// compute segment area within a window
	int AreaIn(const Rect& rc) const
	{
		Rect interRect = rc & box;	// intersection
		if (interRect.area() > 0)
		{
			interRect.x -= box.x;
			interRect.y -= box.y;
			return area_integral_image.Sum(interRect);
			/*float area = cv_area_integral_image.at<float>(interRect.br()) \
				- cv_area_integral_image.at<float>(interRect.br().y, interRect.x) \
				- cv_area_integral_image.at<float>(interRect.y, interRect.br().x) \
				+ cv_area_integral_image.at<float>(interRect.tl());
			return area;*/
		}
		else return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	inline void InitFillArea(int area_in_win)
	{		
		assert(area_in_win > 0);
		leftInnerArea = area_in_win;
		leftOuterArea = extendedArea - leftInnerArea;

		if (leftInnerArea > leftOuterArea)
		{
			leftInnerArea -=  leftOuterArea;
			leftOuterArea = 0;
		}
		else
		{
			leftInnerArea = 0;
			leftOuterArea -= leftInnerArea;
		}

		currentArea = leftInnerArea;
	}

	inline void InitFillArea()
	{
		leftInnerArea = 0;
		leftOuterArea = extendedArea;

		currentArea = 0;
	}

	float dist_to_win;	// distance between centroid to window center

	static bool comp_by_dist(const SegSuperPixelComposeFeature* a, const SegSuperPixelComposeFeature* b)	{	return a->dist_to_win > b->dist_to_win;	}

	static bool comp_by_cost(const SegSuperPixelComposeFeature* a, const SegSuperPixelComposeFeature* b)	{	return a->composition_cost < b->composition_cost;	}

	// x: id, y: area
	static bool comp_by_area(const Point a, const Point b) { return a.y > b.y; }

};


// it holds the key data structure and algorithm for window based composition algorithm
class SegmentBasedWindowComposer
{
public:

	int use_feat;

	vector<SegSuperPixelComposeFeature> sp_comp_features;	// superpixel features

	SegmentBasedWindowComposer();
	~SegmentBasedWindowComposer();

	// input parameters: a rgb image, a segmentation index image
	// 1. compute background weight for each segment
	// 2. compute pair-wise composition cost
	// return whether initialization is done correctly, usually the memory allocation status
	//bool Init(const ImageUIntSimple& seg_map, const vector<SegSuperPixelFeature>& sp_features, const ImageFloatSimple* bg_weight_map = NULL); 

	bool Init(const Mat& seg_map, const vector<SegSuperPixelFeature>& sp_features, const Mat& bg_weight_map = Mat());

	// compute composition cost of a single window
	float Compose(const Rect& win);
	
	// compute composition costs of all windows	in compose_cost_map		
	void ComposeAll(const int win_width, const int win_height, const bool use_ehsw = false);

protected:
	void Clear();  // release internal buffers

	int m_nImgWidth, m_nImgHeight;

	// cost map of a sliding window, of dimension (img_width, img_height)
	// but only dimension (img_width-win_width+1, img_height-win_height+1) is filled
	ImageFloatSimple compose_cost_map;

private:
	//const ImageUIntSimple* seg_index_map;	// input segmentation map, set once in Init()
	Mat seg_index_map;

	vector<SegSuperPixelComposeFeature*> innerSegs;	// segments inside current window
	inline float compose_greedy(const Rect& win);

	float m_fMaxCost; // cost upper bound

#ifdef RECORD_AUXILIARY_INFO
	ImageFloatSimple comp_cost_map;		 // window segment composition cost image
#endif
	
};