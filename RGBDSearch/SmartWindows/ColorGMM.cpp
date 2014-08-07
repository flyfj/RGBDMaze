

#include "ColorGMM.h"


namespace visualsearch
{
	namespace learners
	{
		ColorGMM::ColorGMM( cv::Mat& _model )
		{
			const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
			if( _model.empty() )
			{
				_model.create( 1, modelSize*componentsCount, CV_64FC1 );
				_model.setTo(cv::Scalar(0));
			}
			else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
				CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

			model = _model;

			coefs = model.ptr<double>(0);
			mean = coefs + componentsCount;
			cov = mean + 3*componentsCount;

			for( int ci = 0; ci < componentsCount; ci++ )
				if( coefs[ci] > 0 )
					calcInverseCovAndDeterm( ci );
		}

		double ColorGMM::operator()( const cv::Vec3d color ) const
		{
			double res = 0;
			for( int ci = 0; ci < componentsCount; ci++ )
				res += coefs[ci] * (*this)(ci, color );
			return res;
		}

		double ColorGMM::operator()( int ci, const cv::Vec3d color ) const
		{
			double res = 0;
			if( coefs[ci] > 0 )
			{
				//CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
				cv::Vec3d diff = color;
				double* m = mean + 3*ci;
				diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];	// centralize
				double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
					+ diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
					+ diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
				res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
			}
			return res;
		}

		int ColorGMM::whichComponent( const cv::Vec3d color ) const
		{
			int k = 0;
			double max = 0;

			for( int ci = 0; ci < componentsCount; ci++ )
			{
				double p = (*this)( ci, color );
				if( p > max )
				{
					k = ci;
					max = p;
				}
			}
			return k;
		}

		void ColorGMM::initLearning()
		{
			for( int ci = 0; ci < componentsCount; ci++)
			{
				sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
				prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
				prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
				prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
				sampleCounts[ci] = 0;
			}
			totalSampleCount = 0;
		}

		void ColorGMM::addSample( int ci, const cv::Vec3d color )
		{
			sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
			prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
			prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
			prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
			sampleCounts[ci]++;
			totalSampleCount++;
		}

		void ColorGMM::endLearning()
		{
			const double variance = 0.01;
			for( int ci = 0; ci < componentsCount; ci++ )
			{
				int n = sampleCounts[ci];
				if( n == 0 )
					coefs[ci] = 0;
				else
				{
					coefs[ci] = (double)n/totalSampleCount;

					double* m = mean + 3*ci;
					m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

					double* c = cov + 9*ci;
					c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
					c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
					c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

					double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
					if( dtrm <= std::numeric_limits<double>::epsilon() )
					{
						// Adds the white noise to avoid singular covariance matrix.
						c[0] += variance;
						c[4] += variance;
						c[8] += variance;
					}

					calcInverseCovAndDeterm(ci);
				}
			}
		}

		void ColorGMM::calcInverseCovAndDeterm( int ci )
		{
			if( coefs[ci] > 0 )
			{
				double *c = cov + 9*ci;
				double dtrm =
					covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

				CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
				inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;	// A
				inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;	// B
				inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;	// C
				inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
				inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
				inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
				inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
				inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
				inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
			}
		}


		//////////////////////////////////////////////////////////////////////////

		int GeneralGMM::featureDim = 3;

		GeneralGMM::GeneralGMM( int _featureDim )
		{
			featureDim = _featureDim;
			// compress all component parameter into a one-row matrix

			const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
			model.create( 1, modelSize*componentsCount, CV_64FC1 );
			model.setTo(cv::Scalar(0));

			coefs = model.ptr<double>(0);
			mean = coefs + componentsCount;
			cov = mean + 3*componentsCount;

			for( int ci = 0; ci < componentsCount; ci++ )
				if( coefs[ci] > 0 )
					calcInverseCovAndDeterm( ci );

			return;

			// init
			means.resize(componentsCount);
			covs.resize(componentsCount);
			inv_covs.resize(componentsCount);
			weights.resize(componentsCount);
			covDets.resize(componentsCount);
			for(int ci=0; ci<componentsCount; ci++)
			{
				means[ci].create(1, featureDim, CV_64F);
				means[ci].setTo(0);
				covs[ci].create(featureDim, featureDim, CV_64F);
				covs[ci].setTo(0);
				inv_covs[ci].create(featureDim, featureDim, CV_64F);
				inv_covs[ci].setTo(0);
				weights[ci] = 0;
				covDets[ci] = 0;
			}
		}

		double GeneralGMM::operator()( const cv::Mat& samp ) const
		{
			double res = 0;
			for( int ci = 0; ci < componentsCount; ci++ )
				res += coefs[ci] * (*this)(ci, samp );
			return res;
		}

		double GeneralGMM::operator()( int ci, const cv::Mat& samp ) const
		{
			double res = 0;
			if( coefs[ci] > 0 )
			{
				//CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
				cv::Vec3d diff(samp.at<double>(0), samp.at<double>(1), samp.at<double>(2));
				double* m = mean + 3*ci;
				diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];	// centralize
				double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
					+ diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
					+ diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
				res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
			}
			return res;

			//double res = 0;
			//if( weights[ci] > 0 )
			//{
			//	//CV_Assert( covDets[ci] > std::numeric_limits<double>::epsilon() );

			//	cv::Mat diff = samp - means[ci];
			//	double val = 0;
			//	for(int i=0; i<diff.cols; i++)
			//	{
			//		double cur_sum = 0;
			//		for(int j=0; j<diff.cols; j++)
			//		{
			//			cur_sum += inv_covs[ci].at<double>(i, j) * diff.at<double>(0, j);
			//		}
			//		val += diff.at<double>(0, i) * cur_sum;
			//	}

			//	//cv::Mat val = - diff * inv_covs[ci] * diff.t() * 0.5f;
			//	//CV_Assert( val.rows == 1 && val.cols == 1 );val.at<double>(0,0)
			//
			//	res = 1.0f / sqrt(covDets[ci]) * exp(-val*0.5f);
			//}
			//return res;
		}

		int GeneralGMM::whichComponent( const cv::Mat& samp ) const
		{
			int k = 0;
			double max = 0;

			for( int ci = 0; ci < componentsCount; ci++ )
			{
				double p = (*this)( ci, samp );
				if( p > max )
				{
					k = ci;
					max = p;
				}
			}
			return k;
		}

		void GeneralGMM::initLearning()
		{
			for( int ci = 0; ci < componentsCount; ci++)
			{
				sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
				prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
				prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
				prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
				sampleCounts[ci] = 0;
			}
			totalSampleCount = 0;

			return;

			for( int ci = 0; ci < componentsCount; ci++)
			{
				means[ci].setTo(0);
				covs[ci].setTo(0);

				// reset each component
				sampleCounts[ci] = 0;
			}
			totalSampleCount = 0;
		}

		void GeneralGMM::addSample( int ci, const cv::Mat& samp )
		{
			cv::Vec3d color(samp.at<double>(0), samp.at<double>(1), samp.at<double>(2));
			sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
			prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
			prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
			prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
			sampleCounts[ci]++;
			totalSampleCount++;

			return;

			means[ci] += samp;
			covs[ci] += samp.t() * samp;

			sampleCounts[ci]++;
			totalSampleCount++;
		}

		void GeneralGMM::endLearning()
		{
			const double variance = 0.01;
			for( int ci = 0; ci < componentsCount; ci++ )
			{
				int n = sampleCounts[ci];
				if( n == 0 )
					coefs[ci] = 0;
				else
				{
					coefs[ci] = (double)n/totalSampleCount;

					double* m = mean + 3*ci;
					m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

					double* c = cov + 9*ci;
					c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
					c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
					c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

					double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
					if( dtrm <= std::numeric_limits<double>::epsilon() )
					{
						// Adds the white noise to avoid singular covariance matrix.
						c[0] += variance;
						c[4] += variance;
						c[8] += variance;
					}

					calcInverseCovAndDeterm(ci);
				}
			}

			//const double variance = 0.01;
			//for( int ci = 0; ci < componentsCount; ci++ )
			//{
			//	int n = sampleCounts[ci];
			//	if( n == 0 )
			//		weights[ci] = 0;
			//	else
			//	{
			//		weights[ci] = (double)n/totalSampleCount;

			//		means[ci] /= n;

			//		// covariance (assume diagonal) cov = 1/n sum(X^T X) - mu^T mu
			//		covs[ci] = covs[ci] / n - means[ci].t() * means[ci];
			//		//cout<<covs[ci]<<endl;

			//		// compute determinant
			//		covDets[ci] = cv::determinant(covs[ci]);
			//		if( covDets[ci] <= std::numeric_limits<double>::epsilon() )
			//		{
			//			// Adds the white noise to diagonal to avoid singular covariance matrix.
			//			for(int i=0; i<featureDim; i++)
			//				covs[ci].at<double>(i, i) += variance;
			//		}

			//		calcInverseCovAndDeterm(ci);
			//	}
			//}
		}

		void GeneralGMM::calcInverseCovAndDeterm( int ci )
		{
			if( coefs[ci] > 0 )
			{
				double *c = cov + 9*ci;
				double dtrm =
					covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

				CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
				inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;	// A
				inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;	// B
				inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;	// C
				inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
				inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
				inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
				inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
				inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
				inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
			}

			return;

			if( weights[ci] > 0 )
			{
				cv::invert(covs[ci], inv_covs[ci]);
				//cout<<inv_covs[ci]<<endl;
			}
		}

	}
}


