

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
				CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
				cv::Vec3d diff = color;
				double* m = mean + 3*ci;
				diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
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
				inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
				inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
				inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
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

		GeneralGMM::GeneralGMM( cv::Mat& _model, int _featureDim )
		{
			featureDim = _featureDim;
			// compress all component parameter into a one-row matrix
			const int modelSize = featureDim/*mean*/ + featureDim*featureDim/*covariance*/ + 1/*component weight*/;
			if( _model.empty() )
			{
				_model.create( 1, modelSize*componentsCount, CV_64FC1 );
				_model.setTo(cv::Scalar(0));
			}
			else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
				CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

			model = _model;

			// format: component_weights + means + covs
			// each points to the beginning of the data
			coefs = model.ptr<double>(0);
			mean = coefs + componentsCount;
			cov = mean + featureDim*componentsCount;

			for( int ci = 0; ci < componentsCount; ci++ )
				if( coefs[ci] > 0 )
					calcInverseCovAndDeterm( ci );
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
				CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
				cv::Mat diff = samp;
				double* m = mean + featureDim*ci;
				// mean difference
				for(int i=0; i<featureDim; i++)
					diff.at<double>(i) -= m[i];
				// 
				double mult = 0;
				for(int i=0; i<featureDim; i++)
				{
					double sumv = 0;
					for(int j=0; j<featureDim; j++)
					{
						sumv += diff.at<double>(j)*inverseCovs[ci][j][i];
					}
					mult += diff.at<double>(i)*sumv;
				}
				res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
			}
			return res;
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
				// reset each component
				for(int i=0; i<featureDim; i++)
				{
					sums[ci][i] = 0;
					for(int j=0; j<featureDim; j++)
						prods[ci][i][j] = 0;
				}
				sampleCounts[ci] = 0;
			}
			totalSampleCount = 0;
		}

		void GeneralGMM::addSample( int ci, const cv::Mat& samp )
		{
			for(int i=0; i<featureDim; i++)
			{
				sums[ci][i] += samp.at<double>(i);
				for(int j=0; j<featureDim; j++)
				{
					prods[ci][i][j] += samp.at<double>(i) * samp.at<double>(j);
				}
			}
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

					double* m = mean + featureDim*ci;
					// compute mean
					for(int i=0; i<featureDim; i++)
						m[i] = sums[ci][i] / n;

					// cov pointer
					double* c = cov + featureDim*featureDim*ci;
					cv::Mat cov_mat(featureDim, featureDim, CV_64F);
					for(int i=0; i<featureDim; i++)
					{
						for(int j=0; j<featureDim; j++)
						{
							c[i*featureDim+j] = prods[ci][i][j] / n - m[i]*m[j];
							cov_mat.at<double>(i, j) = c[i*featureDim+j];
						}
					}

					// compute determinant
					double dtrm = cv::determinant(cov_mat);
					if( dtrm <= std::numeric_limits<double>::epsilon() )
					{
						// Adds the white noise to diagonal to avoid singular covariance matrix.
						for(int i=0; i<featureDim; i++)
							cov_mat.at<double>(i, i) += variance;
					}

					calcInverseCovAndDeterm(ci);
				}
			}
		}

		void GeneralGMM::calcInverseCovAndDeterm( int ci )
		{
			if( coefs[ci] > 0 )
			{
				double *c = cov + featureDim*featureDim*ci;
				cv::Mat cov_mat(featureDim, featureDim, CV_64F);
				for(int i=0; i<featureDim; i++)
				{
					for(int j=0; j<featureDim; j++)
					{
						cov_mat.at<double>(i, j) = c[i*featureDim+j];
					}
				}
				double dtrm = covDeterms[ci] = cv::determinant(cov_mat);

				CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
				// compute inverse
				cv::Mat inv_cov;
				cv::invert(cov_mat, inv_cov);

				// put into array
				for(int r=0; r<featureDim; r++)
				{
					for(int c=0; c<featureDim; c++)
						inverseCovs[ci][c][r] = inv_cov.at<double>(r,c);
				}

			}
		}

	}
}


