//////////////////////////////////////////////////////////////////////////
// a modified version of opencv grabcut
// jiefeng@2013-11-07
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common_libs.h"
#include "ColorGMM.h"
#include "gcgraph.hpp"

namespace visualsearch
{

	enum GrabCutType
	{
		GC_RGB,
		GC_RGBD
	};

	// what to use for data and smooth term
	enum GrabCutConfig
	{
		GC_DATA_RGB,
		GC_DATA_DEPTH,
		GC_DATA_RGBD,
		GC_SMOOTH_RGB,
		GC_SMOOTH_DEPTH,
		GC_SMOOTH_RGBD
	};


	// class for grabcut algorithm
	// support rgbd, depth in float
	class GrabCutter
	{
	private:

		learners::GeneralGMM bgdGGMM;
		learners::GeneralGMM fgdGGMM;

		cv::Mat ConvertVec2Mat(const cv::Vec3d color)
		{
			cv::Mat val_mat(1, 3, CV_64F);
			val_mat.at<double>(0,0) = color.val[0];
			val_mat.at<double>(0,1) = color.val[1];
			val_mat.at<double>(0,2) = color.val[2];

			return val_mat;
		}

		cv::Mat ConvertVec2Mat(double x1, double x2, double x3, double x4)
		{
			cv::Mat val_mat(1, 4, CV_64F);
			val_mat.at<double>(0,0) = x1;
			val_mat.at<double>(0,1) = x2;
			val_mat.at<double>(0,2) = x3;
			val_mat.at<double>(0,3) = x4;

			return val_mat;
		}

	public:

		GrabCutConfig DATA_CONFIG;
		GrabCutConfig SMOOTH_CONFIG;

		GrabCutter()
		{
			// use classic rgb
		     DATA_CONFIG = GC_DATA_RGB;
			 SMOOTH_CONFIG = GC_SMOOTH_RGB;
		}

		/*
		Calculate beta - parameter of GrabCut algorithm.
		beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
		*/
		double calcBeta( const cv::Mat& img );

		// rgbd version
		double calcBetaRGBD( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask );

		/*
		Calculate weights of non-terminal vertices of graph.
		beta and gamma - parameters of GrabCut algorithm.
		neighbor weights
		*/

		void calcNWeights( const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma );

		void calcNWeightsRGBD( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, 
			cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma );

		/*
		Check size, type and element values of mask matrix.
		*/
		void checkMask( const cv::Mat& img, const cv::Mat& mask );

		/*
		Initialize mask using rectangular.
		outside -> bg; inside -> pr_fg
		*/
		void initMaskWithRect( cv::Mat& mask, cv::Size imgSize, cv::Rect rect );

		/*
		Initialize GMM background and foreground models using kmeans algorithm.
		*/

		void initGMMs( const cv::Mat& img, const cv::Mat& mask, learners::GeneralGMM& bgdGMM, learners::GeneralGMM& fgdGMM );

		void initRGBDGMMs( const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& mask, learners::GeneralGMM& bgdGMM, learners::GeneralGMM& fgdGMM );

		/*
		Assign GMMs components for each pixel.
		*/

		void assignGMMsComponents( const cv::Mat& img, const cv::Mat& mask, const learners::GeneralGMM& bgdGMM, const learners::GeneralGMM& fgdGMM, cv::Mat& compIdxs );

		void assignRGBDGMMsComponents( const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& mask, const learners::GeneralGMM& bgdGMM, const learners::GeneralGMM& fgdGMM, cv::Mat& compIdxs );

		/*
		Learn GMMs parameters.
		*/

		void learnGMMs( const cv::Mat& img, const cv::Mat& mask, const cv::Mat& compIdxs, learners::GeneralGMM& bgdGMM, learners::GeneralGMM& fgdGMM );

		void learnRGBDGMMs( const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& mask, const cv::Mat& compIdxs, learners::GeneralGMM& bgdGMM, learners::GeneralGMM& fgdGMM );

		/*
		Construct GCGraph
		*/

		void constructGCGraph( const cv::Mat& img, const cv::Mat& mask, const learners::GeneralGMM& bgdGMM, const learners::GeneralGMM& fgdGMM, double lambda,
			const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
			GCGraph<double>& graph );

		void constructRGBDGCGraph( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& mask, const learners::GeneralGMM& bgdGMM, const learners::GeneralGMM& fgdGMM, double lambda,
			const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
			GCGraph<double>& graph );

		/*
		Estimate segmentation using MaxFlow algorithm
		*/
		void estimateSegmentation( GCGraph<double>& graph, cv::Mat& mask );


		//////////////////////////////////////////////////////////////////////////
		// augmented methods

		// predict label of fg / bg using existing models; outside box is bg
		bool predictMask(const cv::Mat& color_img, cv::Mat& mask, const cv::Rect& box, bool show = false);

		bool predictMask(const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& mask, const cv::Rect& box, bool show = false);

		//////////////////////////////////////////////////////////////////////////

		// run grabcut
		// mask is a 0/1 hard map
		bool RunGrabCut( const cv::Mat& img, cv::Mat& fg_mask, const cv::Rect& rect,
			cv::Mat& bgdModel, cv::Mat& fgdModel,
			int iterCount, int mode );

		// rgbd version
		bool RunRGBDGrabCut( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask, const cv::Rect& rect,
			cv::Mat& bgdModel, cv::Mat& fgdModel,
			int iterCount, int mode );

	};
}


