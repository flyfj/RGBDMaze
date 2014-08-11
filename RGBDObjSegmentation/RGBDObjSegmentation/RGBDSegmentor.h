//////////////////////////////////////////////////////////////////////////
// perform iterative saliency segmentation
// jiefeng @ 2012.12.11
//////////////////////////////////////////////////////////////////////////


#pragma once


#include "common.h"
#include "ImageSegmentor.h"
#include "SaliencyComputer.h"
#include <iostream>
#include <set>
#include <map>
#include <algorithm>
using namespace std;
using namespace cv;


namespace Saliency
{

	class SaliencySegmentor: public Mat, protected Rect
	{

	private:

		vector<SegSuperPixelFeature> sp_features;
		Mat lab_img;
		int prim_seg_num;	// primitive superpixel number
		vector<bool> bg_sign;

		// processor
		ImageSegmentor img_segmentor;
		SaliencyComputer sal_computer;

	public:

		SaliencySegmentor(void);
		~SaliencySegmentor(void);

		const Mat& GetSegmentImage() { return img_segmentor.m_segImg; }
		int GetSegIdByLocation(Point pt) { return img_segmentor.m_idxImg.at<int>(pt.y, pt.x); }

		void Init(const Mat& img);

		// simply combine two segments and create a new one with updated segment data; return distance of the two segments
		float MergeSegments(
			const SegSuperPixelFeature& in_seg1, const SegSuperPixelFeature& in_seg2, 
			SegSuperPixelFeature& out_seg, bool onlyCombineFeat = false);

		// do iterative merging to find salient object
		bool MineSalientObjectFromSegment(const Mat& img, int start_seg_id, float& best_saliency);

		bool MineSalientObjectsByMergingPairs(const Mat& img);

		bool SegmentSaliencyMeasure(const Mat& img);

		float SegmentDissimilarity(const SegSuperPixelFeature& seg1, const SegSuperPixelFeature& seg2);

		bool ComputeSaliencyMap(const Mat& img, Mat& sal_map);

		bool ComputeSaliencyMapByBGPropagation(const Mat& img, Mat& sal_map);

	};
}


