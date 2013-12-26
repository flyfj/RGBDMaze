#include "GrabCutter.h"


namespace visualsearch
{

	double GrabCutter::calcBetaRGBD( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask )
	{
		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		// compute expectation of color difference between two neighboring pixels
		double beta = 0;
		for( int y = 0; y < img.rows; y++ )
		{
			for( int x = 0; x < img.cols; x++ )
			{
				cv::Vec3d color_val = (cv::Vec3d)img.at<cv::Vec3b>(y,x);

				if( useDepth && dmask.at<uchar>(y, x) <= 0)
					continue;

				if( x>0 ) // left
				{
					if(useDepth)
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y, x-1);
						beta += diff*diff;
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y, x-1);
						beta += diff.dot(diff);
					}
				}
				if( y>0 && x>0 ) // upleft
				{
					if(useDepth)
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y-1, x-1);
						beta += diff*diff;
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y-1, x-1);
						beta += diff.dot(diff);
					}
				}
				if( y>0 ) // up
				{
					if(useDepth)
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y-1,x);
						beta += diff*diff;
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y-1, x);
						beta += diff.dot(diff);
					}
				}
				if( y>0 && x<dmap.cols-1) // upright
				{
					if(useDepth)
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y-1, x+1);
						beta += diff*diff;
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y-1, x+1);
						beta += diff.dot(diff);
					}
				}
			}
		}
		if( beta <= std::numeric_limits<double>::epsilon() )
			beta = 0;
		else
		{
			if(useDepth)
				beta = 1.f / (2 * beta / cv::countNonZero(dmask));
			else
				beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2));
		}

		return beta;
	}

	//////////////////////////////////////////////////////////////////////////

	void GrabCutter::calcNWeightsRGBD( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask,
		cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma )
	{
		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
		leftW.create( img.rows, img.cols, CV_64FC1 );
		upleftW.create( img.rows, img.cols, CV_64FC1 );
		upW.create( img.rows, img.cols, CV_64FC1 );
		uprightW.create( img.rows, img.cols, CV_64FC1 );
		for( int y = 0; y < img.rows; y++ )
		{
			for( int x = 0; x < img.cols; x++ )
			{
				if( useDepth && dmask.at<uchar>(y,x) <= 0 )
					continue;

				cv::Vec3d color_val = img.at<cv::Vec3b>(y,x);

				if( x-1>=0 ) // left
				{
					if( useDepth )
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y, x-1);
						leftW.at<double>(y,x) = gamma * exp(-beta*diff*diff);
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y, x-1);
						leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
					}		
				}
				else
					leftW.at<double>(y,x) = 0;

				if( x-1>=0 && y-1>=0 ) // upleft
				{
					if( useDepth )
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y-1,x-1);
						upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff*diff);
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y-1, x-1);
						upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
					}
				}
				else
					upleftW.at<double>(y,x) = 0;

				if( y-1>=0 ) // up
				{
					if( useDepth )
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y-1,x);
						upW.at<double>(y,x) = gamma * exp(-beta*diff*diff);
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y-1, x);
						upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
					}
				}
				else
					upW.at<double>(y,x) = 0;

				if( x+1<dmap.cols && y-1>=0 ) // upright
				{
					if( useDepth )
					{
						float diff = dmap.at<float>(y, x) - dmap.at<float>(y-1, x+1);
						uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff*diff);
					}
					else
					{
						cv::Vec3d diff = color_val - (cv::Vec3d)img.at<cv::Vec3b>(y-1, x+1);
						uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
					}
				}
				else
					uprightW.at<double>(y,x) = 0;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////

	void GrabCutter::checkMask( const cv::Mat& img, const cv::Mat& mask )
	{
		if( mask.empty() )
			CV_Error( CV_StsBadArg, "mask is empty" );
		if( mask.type() != CV_8UC1 )
			CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
		if( mask.cols != img.cols || mask.rows != img.rows )
			CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
		for( int y = 0; y < mask.rows; y++ )
		{
			for( int x = 0; x < mask.cols; x++ )
			{
				uchar val = mask.at<uchar>(y,x);
				if( val!=cv::GC_BGD && val!=cv::GC_FGD && val!=cv::GC_PR_BGD && val!=cv::GC_PR_FGD )
					CV_Error( CV_StsBadArg, "mask element value must be equel"
						"GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
			}
		}
	}

	void GrabCutter::initMaskWithRect( cv::Mat& mask, cv::Size imgSize, cv::Rect rect )
	{
		mask.create( imgSize, CV_8UC1 );
		mask.setTo( cv::GC_BGD );

		rect.x = max(0, rect.x);
		rect.y = max(0, rect.y);
		rect.width = min(rect.width, imgSize.width-rect.x);
		rect.height = min(rect.height, imgSize.height-rect.y);

		(mask(rect)).setTo( cv::Scalar( cv::GC_PR_FGD ) );
	}

	//////////////////////////////////////////////////////////////////////////


	void GrabCutter::initRGBDGMMs( const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& fg_mask )
	{
		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		const int kMeansItCount = 10;
		const int kMeansType = cv::KMEANS_PP_CENTERS;

		cv::Mat bgdLabels, fgdLabels;
		cv::Mat bgdSamples(0, 0, CV_64F), fgdSamples(0, 0, CV_64F);
		cv::Point p;
		// separate bg and fg samples
		for( p.y = 0; p.y < color_img.rows; p.y++ )
		{
			for( p.x = 0; p.x < color_img.cols; p.x++ )
			{
				cv::Vec3f colorval = (cv::Vec3f)color_img.at<cv::Vec3b>(p);
				cv::Mat samp;
				if( useDepth )
				{
					if(dmask.at<uchar>(p) <= 0)
						continue;
					float dval = dmap.at<float>(p);

					samp = ConvertVec2Mat( colorval.val[0], colorval.val[1], colorval.val[2], dval );
				}
				else
				{
					samp = ConvertVec2Mat( colorval.val[0], colorval.val[1], colorval.val[2] );
				}

				if( fg_mask.at<uchar>(p) == cv::GC_BGD || fg_mask.at<uchar>(p) == cv::GC_PR_BGD )
					bgdSamples.push_back( samp );
				else // GC_FGD | GC_PR_FGD
					fgdSamples.push_back( samp );
			}
		}
		CV_Assert( !bgdSamples.empty() && !bgdSamples.empty() );

		cv::Mat _bgdSamples;
		bgdSamples.convertTo(_bgdSamples, CV_32F);
		kmeans( _bgdSamples, learners::GeneralGMM::componentsCount, bgdLabels,
			cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
		cv::Mat _fgdSamples;
		fgdSamples.convertTo(_fgdSamples, CV_32F);
		kmeans( _fgdSamples, learners::GeneralGMM::componentsCount, fgdLabels,
			cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

		bgdGGMM.initLearning();
		for( int i = 0; i < bgdSamples.rows; i++ )
			bgdGGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples.row(i) );
		bgdGGMM.endLearning();

		fgdGGMM.initLearning();
		for( int i = 0; i < fgdSamples.rows; i++ )
			fgdGGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples.row(i) );
		fgdGGMM.endLearning();
	}

	//////////////////////////////////////////////////////////////////////////

	void GrabCutter::assignRGBDGMMsComponents( const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& fg_mask, cv::Mat& compIdxs )
	{
		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		cv::Point p;
		for( p.y = 0; p.y < color_img.rows; p.y++ )
		{
			for( p.x = 0; p.x < color_img.cols; p.x++ )
			{
				cv::Vec3d color = (cv::Vec3d)color_img.at<cv::Vec3b>(p);
				cv::Mat samp;

				if( useDepth )
				{
					if( dmask.at<uchar>(p) <= 0 )
						continue;

					float dval = dmap.at<float>(p);
					samp = ConvertVec2Mat(color.val[0], color.val[1], color.val[2], dval);
				}
				else
					samp = ConvertVec2Mat(color.val[0], color.val[1], color.val[2]);
				
				compIdxs.at<int>(p) = fg_mask.at<uchar>(p) == cv::GC_BGD || fg_mask.at<uchar>(p) == cv::GC_PR_BGD ?
					bgdGGMM.whichComponent(samp) : fgdGGMM.whichComponent(samp);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////

	void GrabCutter::learnRGBDGMMs( const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& fg_mask, const cv::Mat& compIdxs )
	{
		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		bgdGGMM.initLearning();
		fgdGGMM.initLearning();
		cv::Point p;
		for( int ci = 0; ci < learners::GeneralGMM::componentsCount; ci++ )
		{
			for( p.y = 0; p.y < color_img.rows; p.y++ )
			{
				for( p.x = 0; p.x < color_img.cols; p.x++ )
				{
					if( compIdxs.at<int>(p) == ci )
					{
						cv::Vec3d color = (cv::Vec3d)color_img.at<cv::Vec3b>(p);
						cv::Mat samp;

						if( useDepth )
						{
							// rgbd
							if( dmask.at<uchar>(p) <= 0 )
								continue;

							float dval = dmap.at<float>(p);
							samp = ConvertVec2Mat(color.val[0], color.val[1], color.val[2], dval);
						}
						else
						{
							samp = ConvertVec2Mat(color.val[0], color.val[1], color.val[2]);
						}
						
						if( fg_mask.at<uchar>(p) == cv::GC_BGD || fg_mask.at<uchar>(p) == cv::GC_PR_BGD )
							bgdGGMM.addSample( ci, samp );
						else
							fgdGGMM.addSample( ci, samp );
					}
				}
			}
		}

		bgdGGMM.endLearning();
		fgdGGMM.endLearning();
	}

	//////////////////////////////////////////////////////////////////////////


	void GrabCutter::constructRGBDGCGraph( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, const cv::Mat& fg_mask, double lambda,
		const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
		GCGraph<double>& graph )
	{
		// check consistency
		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		int vtxCount = img.cols*img.rows,
			edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
		graph.create(vtxCount, edgeCount);
		cv::Point p;
		for( p.y = 0; p.y < img.rows; p.y++ )
		{
			for( p.x = 0; p.x < img.cols; p.x++ )
			{
				// add node
				int vtxIdx = graph.addVtx();
				cv::Vec3d color = (cv::Vec3d)img.at<cv::Vec3b>(p);

				cv::Mat samp;
				if( !useDepth )
					samp = ConvertVec2Mat(color.val[0], color.val[1], color.val[2]);
				else
					samp = ConvertVec2Mat(color.val[0], color.val[1], color.val[2], dmap.at<double>(p));

				// set t-weights
				// source: bg; sink: fg
				double fromSource, toSink;
				if( fg_mask.at<uchar>(p) == cv::GC_PR_BGD || fg_mask.at<uchar>(p) == cv::GC_PR_FGD )
				{
					fromSource = -log( bgdGGMM(samp) );
					toSink = -log( fgdGGMM(samp) );
				}
				else if( fg_mask.at<uchar>(p) == cv::GC_BGD )
				{
					fromSource = 0;
					toSink = lambda;
				}
				else // GC_FGD
				{
					fromSource = lambda;
					toSink = 0;
				}
				graph.addTermWeights( vtxIdx, fromSource, toSink );

				// set n-weights
				if( p.x>0 )
				{
					double w = leftW.at<double>(p);
					graph.addEdges( vtxIdx, vtxIdx-1, w, w );
				}
				if( p.x>0 && p.y>0 )
				{
					double w = upleftW.at<double>(p);
					graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
				}
				if( p.y>0 )
				{
					double w = upW.at<double>(p);
					graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
				}
				if( p.x<img.cols-1 && p.y>0 )
				{
					double w = uprightW.at<double>(p);
					graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
				}
			}
		}
	} 

	//////////////////////////////////////////////////////////////////////////

	void GrabCutter::estimateSegmentation( GCGraph<double>& graph, cv::Mat& mask )
	{
		graph.maxFlow();
		cv::Point p;
		for( p.y = 0; p.y < mask.rows; p.y++ )
		{
			for( p.x = 0; p.x < mask.cols; p.x++ )
			{
				if( mask.at<uchar>(p) == cv::GC_PR_BGD || mask.at<uchar>(p) == cv::GC_PR_FGD )
				{
					if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
						mask.at<uchar>(p) = cv::GC_PR_FGD;
					else
						mask.at<uchar>(p) = cv::GC_PR_BGD;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////

	bool GrabCutter::predictMask(const cv::Mat& color_img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& mask, const cv::Rect& box, bool show)
	{
		if( !dmap.empty() && !dmask.empty() && bgdGGMM.featureDim != 4 )
		{
			std::cerr<<"Inconsistent model format for rgbd mode."<<std::endl;
			return false;
		}

		bool useDepth = (!dmap.empty() && !dmask.empty() ? true: false);

		mask.create(color_img.rows, color_img.cols, CV_8U);
		mask.setTo(cv::GC_BGD);

		cv::Vec3b redcolor(0,0,255);
		cv::Vec3b bluecolor(255,0,0);
		cv::Mat disp_mask(color_img.rows, color_img.cols, CV_8UC3);
		disp_mask.setTo(cv::Vec3b(0,255,0));

		// predict for pixels inside rect
		if( dmap.empty() || dmask.empty() )
		{
			// rgb mode
			for(int r=box.y; r<box.br().y; r++)
			{
				for(int c=box.x; c<box.br().x; c++)
				{
					cv::Vec3d cur_color = (cv::Vec3d)color_img.at<cv::Vec3b>(r,c);
					double bg_prob = bgdGGMM( ConvertVec2Mat( cur_color.val[0], cur_color.val[1], cur_color.val[2] ) );
					double fg_prob = fgdGGMM( ConvertVec2Mat( cur_color.val[0], cur_color.val[1], cur_color.val[2] ) );
					mask.at<uchar>(r,c) = (bg_prob > fg_prob? cv::GC_PR_BGD: cv::GC_PR_FGD);
					disp_mask.at<cv::Vec3b>(r,c) = (bg_prob > fg_prob? bluecolor: redcolor);
				}
			}
		}
		else
		{
			// rgbd
			for(int r=box.y; r<box.br().y; r++)
			{
				for(int c=box.x; c<box.br().x; c++)
				{
					if( dmask.at<uchar>(r,c) > 0 )
					{
						cv::Vec3d cur_color = (cv::Vec3d)color_img.at<cv::Vec3b>(r,c);
						double bg_prob = bgdGGMM( ConvertVec2Mat(cur_color.val[0], cur_color.val[1], cur_color.val[2], dmap.at<float>(r,c)) );
						double fg_prob = fgdGGMM( ConvertVec2Mat(cur_color.val[0], cur_color.val[1], cur_color.val[2], dmap.at<float>(r,c)) );
						mask.at<uchar>(r,c) = (bg_prob > fg_prob? cv::GC_PR_BGD: cv::GC_PR_FGD);
						disp_mask.at<cv::Vec3b>(r,c) = (bg_prob > fg_prob? bluecolor: redcolor);
					}
				}
			}
		}
		

		if( show )
		{
			// visualize
			cv::imshow("pred_mask", disp_mask);
			cv::waitKey(10);
		}

		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	

	bool GrabCutter::RunRGBDGrabCut( const cv::Mat& img, const cv::Mat& dmap, const cv::Mat& dmask, cv::Mat& fg_mask, 
		const cv::Rect& rect, int iterCount, GrabCutMode mode )
	{
		if( img.type() != CV_8UC3 )
		{
			std::cerr<<"image mush have CV_8UC3 type"<<std::endl;
			return false;
		}
		if( !dmap.empty() && dmap.type() != CV_32F )
		{
			std::cerr<<"depth map must have CV_32F type"<<std::endl;
			return false;
		}
		

		if(dmap.empty() && dmask.empty())
		{
			bgdGGMM = learners::GeneralGMM( 3 );
			fgdGGMM = learners::GeneralGMM( 3 );
		}
		else
		{
			bgdGGMM = learners::GeneralGMM( 4 );
			fgdGGMM = learners::GeneralGMM( 4 );
		}
		
		cv::Mat compIdxs( img.size(), CV_32SC1 );

		if( mode == GC_MODE_NEW )
		{
			initMaskWithRect( fg_mask, img.size(), rect );
			//checkMask( img, fg_mask );

			initRGBDGMMs( img, dmap, dmask, fg_mask );
		}

		if( iterCount <= 0)
			return false;

		if( mode == GC_MODE_CONT )
			checkMask( img, fg_mask );

		const double gamma = 50;
		const double lambda = 9*gamma;
		double beta = 1;
		if(SMOOTH_CONFIG == GC_SMOOTH_DEPTH)
			beta = calcBetaRGBD( img, dmap, dmask );
		if(SMOOTH_CONFIG == GC_SMOOTH_RGB)
			beta = calcBetaRGBD( img, cv::Mat(), cv::Mat() );

		cv::Mat leftW, upleftW, upW, uprightW;
		if(SMOOTH_CONFIG == GC_SMOOTH_DEPTH)
			calcNWeightsRGBD(img, dmap, dmask, leftW, upleftW, upW, uprightW, beta, gamma);
		if(SMOOTH_CONFIG == GC_SMOOTH_RGB)
			calcNWeightsRGBD(img, cv::Mat(), cv::Mat(), leftW, upleftW, upW, uprightW, beta, gamma);

		for( int i = 0; i < iterCount; i++ )
		{
			GCGraph<double> graph;
			// assign each pixel to one of the component based on new mask
			assignRGBDGMMsComponents( img, dmap, dmask, fg_mask, compIdxs );
			// re-estimate GMM model using the new mask
			learnRGBDGMMs( img, dmap, dmask, fg_mask, compIdxs );
			// do graph-cut
			constructRGBDGCGraph( img, dmap, dmask, fg_mask, lambda, leftW, upleftW, upW, uprightW, graph );
			// do segment prediction
			estimateSegmentation( graph, fg_mask );
		}

		return true;
	}

}

