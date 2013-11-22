//////////////////////////////////////////////////////////////////////////
// color GMM model modified from opencv grabcut code
// jiefeng@2013-11-07
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common_libs.h"


namespace visualsearch
{
	namespace learners
	{
		class ColorGMM
		{
		public:
			static const int componentsCount = 5;

			ColorGMM() {}
			ColorGMM( cv::Mat& _model );
			// probability of color in the whole model (weighted sum)
			double operator()( const cv::Vec3d color ) const;
			// probability of color in ci component
			double operator()( int ci, const cv::Vec3d color ) const;
			int whichComponent( const cv::Vec3d color ) const;

			void initLearning();
			void addSample( int ci, const cv::Vec3d color );
			void endLearning();

		private:
			void calcInverseCovAndDeterm( int ci );
			cv::Mat model;
			double* coefs;
			double* mean;
			double* cov;

			double inverseCovs[componentsCount][3][3];
			double covDeterms[componentsCount];

			double sums[componentsCount][3];
			double prods[componentsCount][3][3];
			int sampleCounts[componentsCount];
			int totalSampleCount;
		};

		//////////////////////////////////////////////////////////////////////////

		// if something is not right, check the data order from matrix conversion to array
		class GeneralGMM
		{
		public:
			static const int componentsCount = 5;
			static int featureDim;
			static const int MaxFeatureDim = 10;

			GeneralGMM() {}
			GeneralGMM( cv::Mat& _model, int _featureDim );
			// probability of color in the whole model (weighted sum)
			double operator()( const cv::Mat& samp ) const;
			// probability of color in ci component
			double operator()( int ci, const cv::Mat& samp ) const;
			int whichComponent( const cv::Mat& samp ) const;

			void initLearning();
			void addSample( int ci, const cv::Mat& samp );
			void endLearning();

		private:
			void calcInverseCovAndDeterm( int ci );
			cv::Mat model;
			double* coefs;
			double* mean;
			double* cov;

			double inverseCovs[componentsCount][MaxFeatureDim][MaxFeatureDim];	// maximum number of feature dimension
			double covDeterms[componentsCount];

			double sums[componentsCount][MaxFeatureDim];	// intermediate sum of samples
			double prods[componentsCount][MaxFeatureDim][MaxFeatureDim];	// intermediate sum of products
			int sampleCounts[componentsCount];
			int totalSampleCount;
		};
	}
}


