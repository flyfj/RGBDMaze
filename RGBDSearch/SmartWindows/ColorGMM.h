//////////////////////////////////////////////////////////////////////////
// color GMM model modified from opencv grabcut code
// jiefeng@2013-11-07
//////////////////////////////////////////////////////////////////////////


#pragma once

#include "common.h"


namespace visualsearch
{
	namespace learners
	{
		// original opencv gmm
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

		struct GMMParams 
		{
			int componentsCount;
			int featureDim;
			bool cov_diag;

			GMMParams()
			{
				componentsCount = 5;
				featureDim = 3;
				cov_diag = false;
			}
		};

		// if something is not right, check the data order from matrix conversion to array
		class GeneralGMM
		{
		public:

			static const int componentsCount = 5;
			static int featureDim;
			static const int MaxFeatureDim = 10;

			GeneralGMM() { GeneralGMM(3); } 
			GeneralGMM( int _featureDim );
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
			
			// model data
			vector<cv::Mat> means;	// mean feature
			vector<cv::Mat> covs;	// covariance matrix
			vector<double> covDets;	// determinant of cov
			vector<cv::Mat> inv_covs;	// pre-computed inv_conv
			vector<double> weights;	// component weights

			int sampleCounts[componentsCount];
			int totalSampleCount;

			// for 3 dimension use
			double* coefs;
			double* mean;
			double* cov;

			double inverseCovs[componentsCount][3][3];
			double covDeterms[componentsCount];

			double sums[componentsCount][3];
			double prods[componentsCount][3][3];
		};
	}
}


