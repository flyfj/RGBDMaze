#include "GrabCutter.h"


namespace visualsearch
{
	
	double GrabCutter::calcBeta( const cv::Mat& img )
	{
		// compute expectation of color difference between two neighboring pixels
		double beta = 0;
		for( int y = 0; y < img.rows; y++ )
		{
			for( int x = 0; x < img.cols; x++ )
			{
				cv::Vec3d color = img.at<cv::Vec3b>(y,x);
				if( x>0 ) // left
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y,x-1);
					beta += diff.dot(diff);
				}
				if( y>0 && x>0 ) // upleft
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x-1);
					beta += diff.dot(diff);
				}
				if( y>0 ) // up
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x);
					beta += diff.dot(diff);
				}
				if( y>0 && x<img.cols-1) // upright
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x+1);
					beta += diff.dot(diff);
				}
			}
		}
		if( beta <= std::numeric_limits<double>::epsilon() )
			beta = 0;
		else
			beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

		return beta;
	}

	double GrabCutter::calcBetaRGBD( const cv::Mat& img, const cv::Mat& dmap )
	{
		// compute expectation of color difference between two neighboring pixels
		double beta = 0;
		for( int y = 0; y < dmap.rows; y++ )
		{
			for( int x = 0; x < dmap.cols; x++ )
			{
				float dval = dmap.at<float>(y,x);
				if( x>0 ) // left
				{
					float diff = dval - dmap.at<float>(y,x-1);
					beta += diff*diff;
				}
				if( y>0 && x>0 ) // upleft
				{
					float diff = dval - dmap.at<float>(y-1,x-1);
					beta += diff*diff;
				}
				if( y>0 ) // up
				{
					float diff = dval - dmap.at<float>(y-1,x);
					beta += diff*diff;
				}
				if( y>0 && x<img.cols-1) // upright
				{
					float diff = dval - dmap.at<float>(y-1,x+1);
					beta += diff*diff;
				}
			}
		}
		if( beta <= std::numeric_limits<double>::epsilon() )
			beta = 0;
		else
			beta = 1.f / (2 * beta/(4*dmap.cols*dmap.rows - 3*dmap.cols - 3*dmap.rows + 2) );

		return beta;
	}

	void GrabCutter::calcNWeights( const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma )
	{
		const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
		leftW.create( img.rows, img.cols, CV_64FC1 );
		upleftW.create( img.rows, img.cols, CV_64FC1 );
		upW.create( img.rows, img.cols, CV_64FC1 );
		uprightW.create( img.rows, img.cols, CV_64FC1 );
		for( int y = 0; y < img.rows; y++ )
		{
			for( int x = 0; x < img.cols; x++ )
			{
				cv::Vec3d color = img.at<cv::Vec3b>(y,x);
				if( x-1>=0 ) // left
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y,x-1);
					leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
				}
				else
					leftW.at<double>(y,x) = 0;
				if( x-1>=0 && y-1>=0 ) // upleft
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x-1);
					upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
				}
				else
					upleftW.at<double>(y,x) = 0;
				if( y-1>=0 ) // up
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x);
					upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
				}
				else
					upW.at<double>(y,x) = 0;
				if( x+1<img.cols && y-1>=0 ) // upright
				{
					cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x+1);
					uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
				}
				else
					uprightW.at<double>(y,x) = 0;
			}
		}
	}

	void GrabCutter::calcNWeightsRGBD( const cv::Mat& img, const cv::Mat& dmap, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma )
	{
		const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
		leftW.create( dmap.rows, dmap.cols, CV_64FC1 );
		upleftW.create( dmap.rows, dmap.cols, CV_64FC1 );
		upW.create( dmap.rows, dmap.cols, CV_64FC1 );
		uprightW.create( dmap.rows, dmap.cols, CV_64FC1 );
		for( int y = 0; y < dmap.rows; y++ )
		{
			for( int x = 0; x < dmap.cols; x++ )
			{
				float dval = dmap.at<float>(y,x);
				if( x-1>=0 ) // left
				{
					float diff = dval - dmap.at<float>(y,x-1);
					leftW.at<double>(y,x) = gamma * exp(-beta*diff*diff);
				}
				else
					leftW.at<double>(y,x) = 0;
				if( x-1>=0 && y-1>=0 ) // upleft
				{
					float diff = dval - dmap.at<float>(y-1,x-1);
					upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff*diff);
				}
				else
					upleftW.at<double>(y,x) = 0;
				if( y-1>=0 ) // up
				{
					float diff = dval - dmap.at<float>(y-1,x);
					upW.at<double>(y,x) = gamma * exp(-beta*diff*diff);
				}
				else
					upW.at<double>(y,x) = 0;
				if( x+1<img.cols && y-1>=0 ) // upright
				{
					float diff = dval - dmap.at<float>(y-1,x+1);
					uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff*diff);
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

	void GrabCutter::initGMMs( const cv::Mat& img, const cv::Mat& mask, learners::ColorGMM& bgdGMM, learners::ColorGMM& fgdGMM )
	{
		const int kMeansItCount = 10;
		const int kMeansType = cv::KMEANS_PP_CENTERS;

		cv::Mat bgdLabels, fgdLabels;
		vector<cv::Vec3f> bgdSamples, fgdSamples;
		cv::Point p;
		// separate bg and fg samples
		for( p.y = 0; p.y < img.rows; p.y++ )
		{
			for( p.x = 0; p.x < img.cols; p.x++ )
			{
				if( mask.at<uchar>(p) == cv::GC_BGD || mask.at<uchar>(p) == cv::GC_PR_BGD )
					bgdSamples.push_back( (cv::Vec3f)img.at<cv::Vec3b>(p) );
				else // GC_FGD | GC_PR_FGD
					fgdSamples.push_back( (cv::Vec3f)img.at<cv::Vec3b>(p) );
			}
		}
		CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );

		cv::Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
		kmeans( _bgdSamples, learners::ColorGMM::componentsCount, bgdLabels,
				cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
		cv::Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
		kmeans( _fgdSamples, learners::ColorGMM::componentsCount, fgdLabels,
				cv::TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

		bgdGMM.initLearning();
		for( int i = 0; i < (int)bgdSamples.size(); i++ )
			bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
		bgdGMM.endLearning();

		fgdGMM.initLearning();
		for( int i = 0; i < (int)fgdSamples.size(); i++ )
			fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
		fgdGMM.endLearning();
	}

	void GrabCutter::assignGMMsComponents( const cv::Mat& img, const cv::Mat& mask, const learners::ColorGMM& bgdGMM, const learners::ColorGMM& fgdGMM, cv::Mat& compIdxs )
	{
		cv::Point p;
		for( p.y = 0; p.y < img.rows; p.y++ )
		{
			for( p.x = 0; p.x < img.cols; p.x++ )
			{
				cv::Vec3d color = img.at<cv::Vec3b>(p);
				compIdxs.at<int>(p) = mask.at<uchar>(p) == cv::GC_BGD || mask.at<uchar>(p) == cv::GC_PR_BGD ?
					bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
			}
		}
	}

	void GrabCutter::learnGMMs( const cv::Mat& img, const cv::Mat& mask, const cv::Mat& compIdxs, learners::ColorGMM& bgdGMM, learners::ColorGMM& fgdGMM )
	{
		bgdGMM.initLearning();
		fgdGMM.initLearning();
		cv::Point p;
		for( int ci = 0; ci < learners::ColorGMM::componentsCount; ci++ )
		{
			for( p.y = 0; p.y < img.rows; p.y++ )
			{
				for( p.x = 0; p.x < img.cols; p.x++ )
				{
					if( compIdxs.at<int>(p) == ci )
					{
						if( mask.at<uchar>(p) == cv::GC_BGD || mask.at<uchar>(p) == cv::GC_PR_BGD )
							bgdGMM.addSample( ci, img.at<cv::Vec3b>(p) );
						else
							fgdGMM.addSample( ci, img.at<cv::Vec3b>(p) );
					}
				}
			}
		}
		bgdGMM.endLearning();
		fgdGMM.endLearning();
	}

	void GrabCutter::constructGCGraph( const cv::Mat& img, const cv::Mat& mask, const learners::ColorGMM& bgdGMM, const learners::ColorGMM& fgdGMM, double lambda,
						   const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
						   GCGraph<double>& graph )
	{
		int vtxCount = img.cols*img.rows,
			edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
		graph.create(vtxCount, edgeCount);
		cv::Point p;
		for( p.y = 0; p.y < img.rows; p.y++ )
		{
			for( p.x = 0; p.x < img.cols; p.x++)
			{
				// add node
				int vtxIdx = graph.addVtx();
				cv::Vec3b color = img.at<cv::Vec3b>(p);

				// set t-weights
				// source: bg; sink: fg
				double fromSource, toSink;
				if( mask.at<uchar>(p) == cv::GC_PR_BGD || mask.at<uchar>(p) == cv::GC_PR_FGD )
				{
					fromSource = -log( bgdGMM(color) );
					toSink = -log( fgdGMM(color) );
				}
				else if( mask.at<uchar>(p) == cv::GC_BGD )
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

	bool GrabCutter::predictMask(const cv::Mat& color_img, cv::Mat& mask, const cv::Rect& box, bool show)
	{
		mask.create(color_img.rows, color_img.cols, CV_8U);
		mask.setTo(cv::GC_BGD);

		cv::Vec3b redcolor(0,0,255);
		cv::Vec3b bluecolor(255,0,0);
		cv::Mat disp_mask(color_img.rows, color_img.cols, CV_8UC3);
		disp_mask.setTo(cv::Vec3b(0,255,0));

		// predict for pixels inside rect
		for(int r=box.y; r<box.br().y; r++)
		{
			for(int c=box.x; c<box.br().x; c++)
			{
				cv::Vec3d cur_color = (cv::Vec3d)color_img.at<cv::Vec3b>(r,c);
				double bg_prob = bgdGMM(cur_color);
				double fg_prob = fgdGMM(cur_color);
				mask.at<uchar>(r,c) = (bg_prob > fg_prob? cv::GC_PR_BGD: cv::GC_PR_FGD);
				disp_mask.at<cv::Vec3b>(r,c) = (bg_prob > fg_prob? bluecolor: redcolor);
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
	
	bool GrabCutter::RunGrabCut( const cv::Mat& img, cv::Mat& mask, const cv::Rect& rect,
		cv::Mat& bgdModel, cv::Mat& fgdModel,
		int iterCount, int mode )
	{
		if( img.empty() )
		{
			std::cerr<<"image is empty"<<std::endl;
			return false;
		}
		if( img.type() != CV_8UC3 )
		{
			std::cerr<<"image mush have CV_8UC3 type"<<std::endl;
			return false;
		}

		bgdGMM = learners::ColorGMM( bgdModel );
		fgdGMM = learners::ColorGMM( fgdModel );
		cv::Mat compIdxs( img.size(), CV_32SC1 );

		if( mode == cv::GC_INIT_WITH_RECT || mode == cv::GC_INIT_WITH_MASK )
		{
			if( mode == cv::GC_INIT_WITH_RECT )
				initMaskWithRect( mask, img.size(), rect );
			else // flag == GC_INIT_WITH_MASK
				checkMask( img, mask );

			initGMMs( img, mask, bgdGMM, fgdGMM );
		}

		if( iterCount <= 0)
			return false;

		if( mode == cv::GC_EVAL )
			checkMask( img, mask );

		const double gamma = 50;
		const double lambda = 9*gamma;
		const double beta = calcBeta( img );

		cv::Mat leftW, upleftW, upW, uprightW;
		calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );
		// to add depth here


		for( int i = 0; i < iterCount; i++ )
		{
			GCGraph<double> graph;
			// assign each pixel to one of the component based on new mask
			assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
			// re-estimate GMM model using the new mask
			learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
			// do graph-cut
			constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
			// do segment prediction
			estimateSegmentation( graph, mask );
		}

		return true;
	}

	bool GrabCutter::RunGrabCut( const cv::Mat& img, const cv::Mat& dmap, cv::Mat& mask, 
		const cv::Rect& rect, cv::Mat& bgdModel, cv::Mat& fgdModel, 
		int iterCount, int mode )
	{
		if( img.empty() || dmap.empty() )
		{
			std::cerr<<"image / dmap is empty"<<std::endl;
			return false;
		}
		if( img.type() != CV_8UC3 )
		{
			std::cerr<<"image mush have CV_8UC3 type"<<std::endl;
			return false;
		}
		if( dmap.type() != CV_32F )
		{
			std::cerr<<"depth map must have CV_32F type"<<std::endl;
			return false;
		}
		

		bgdGMM = learners::ColorGMM( bgdModel );
		fgdGMM = learners::ColorGMM( fgdModel );
		cv::Mat compIdxs( img.size(), CV_32SC1 );

		if( mode == cv::GC_INIT_WITH_RECT || mode == cv::GC_INIT_WITH_MASK )
		{
			if( mode == cv::GC_INIT_WITH_RECT )
				initMaskWithRect( mask, img.size(), rect );
			else // flag == GC_INIT_WITH_MASK
				checkMask( img, mask );

			initGMMs( img, mask, bgdGMM, fgdGMM );
		}

		if( iterCount <= 0)
			return false;

		if( mode == cv::GC_EVAL )
			checkMask( img, mask );

		const double gamma = 50;
		const double lambda = 9*gamma;
		const double beta = calcBetaRGBD(img, dmap); //calcBeta( img );

		cv::Mat leftW, upleftW, upW, uprightW;
		calcNWeightsRGBD(img, dmap, leftW, upleftW, upW, uprightW, beta, gamma);
		//calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );
		// to add depth here


		for( int i = 0; i < iterCount; i++ )
		{
			GCGraph<double> graph;
			// assign each pixel to one of the component based on new mask
			assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
			// re-estimate GMM model using the new mask
			learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
			// do graph-cut
			constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
			// do segment prediction
			estimateSegmentation( graph, mask );
		}

		return true;
	}

}

