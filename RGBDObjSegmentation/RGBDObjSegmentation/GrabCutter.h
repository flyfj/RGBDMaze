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


	// class for grabcut algorithm
	// support rgbd, depth in float
	class GrabCutter
	{
	private:

		learners::ColorGMM bgdGMM;
		learners::ColorGMM fgdGMM;

	public:

		GrabCutter(){}

		/*
		Calculate beta - parameter of GrabCut algorithm.
		beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
		*/
		double calcBeta( const cv::Mat& img );

		// rgbd version
		double calcBetaRGBD( const cv::Mat& img, const cv::Mat& dmap );

		/*
		Calculate weights of non-terminal vertices of graph.
		beta and gamma - parameters of GrabCut algorithm.
		neighbor weights
		*/
		void calcNWeights( const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma );

		void calcNWeightsRGBD( const cv::Mat& img, const cv::Mat& dmap, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma );

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
		void initGMMs( const cv::Mat& img, const cv::Mat& mask, learners::ColorGMM& bgdGMM, learners::ColorGMM& fgdGMM );

		/*
		Assign GMMs components for each pixel.
		*/
		void assignGMMsComponents( const cv::Mat& img, const cv::Mat& mask, const learners::ColorGMM& bgdGMM, const learners::ColorGMM& fgdGMM, cv::Mat& compIdxs );

		/*
		Learn GMMs parameters.
		*/
		void learnGMMs( const cv::Mat& img, const cv::Mat& mask, const cv::Mat& compIdxs, learners::ColorGMM& bgdGMM, learners::ColorGMM& fgdGMM );

		/*
		Construct GCGraph
		*/
		void constructGCGraph( const cv::Mat& img, const cv::Mat& mask, const learners::ColorGMM& bgdGMM, const learners::ColorGMM& fgdGMM, double lambda,
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

		//////////////////////////////////////////////////////////////////////////

		// run grabcut
		// mask is a 0/1 hard map
		bool RunGrabCut( const cv::Mat& img, cv::Mat& mask, const cv::Rect& rect,
			cv::Mat& bgdModel, cv::Mat& fgdModel,
			int iterCount, int mode );


		// rgbd version
		bool RunGrabCut( const cv::Mat& img, const cv::Mat& dmap, cv::Mat& mask, const cv::Rect& rect,
			cv::Mat& bgdModel, cv::Mat& fgdModel,
			int iterCount, int mode );

	};
}


