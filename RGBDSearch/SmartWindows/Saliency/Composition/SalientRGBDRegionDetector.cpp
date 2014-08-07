#include "SalientRGBDRegionDetector.h"


SalientRGBDRegionDetector::SalientRGBDRegionDetector(void)
{
}


//////////////////////////////////////////////////////////////////////////
void SalientRGBDRegionDetector::Clear()
{
	Base::Clear();

	segmentedImg.clear();
	sp_features.clear();
}


bool SalientRGBDRegionDetector::Init(const int stype, const Mat& cimg, const Mat& dmap)
{
	// clear data for new image
	Clear();

	saltype = stype;

	segmentedImg.resize(3*cimg.cols*cimg.rows);

	// run segmentation
	float sigma = g_para.segSigma, c = g_para.segThresholdK; 
	int min_size = g_para.segMinArea;

	Mat segmentMat;
	imgSegmentor.m_dMinArea = min_size;
	imgSegmentor.m_dSmoothSigma = sigma;
	int superpixel_num = imgSegmentor.DoSegmentation(cimg);
	// visualize segment image
	//imshow("segmentimage", imgSegmentor.m_segImg);
	//waitKey(0);

	//////////////////////////////////////////////////////////////////////////
	// compute features for each superpixel
	//////////////////////////////////////////////////////////////////////////
	// create data
	sp_features.resize(superpixel_num);

	// compute perimeter and boundary pixel numbers, set bounding box
	for(int y=0; y<cimg.rows; y++)
	{
		for(int x=0; x<cimg.cols; x++)
		{
			int seg_id = imgSegmentor.m_idxImg.at<int>(y, x);
			// add area
			sp_features[seg_id].area++;
			// add centroid
			sp_features[seg_id].centroid.x += x;
			sp_features[seg_id].centroid.y += y;

			// update bounding box points
			sp_features[seg_id].box_pos[0].x = min(sp_features[seg_id].box_pos[0].x, x);
			sp_features[seg_id].box_pos[0].y = min(sp_features[seg_id].box_pos[0].y, y);
			sp_features[seg_id].box_pos[1].x = max(sp_features[seg_id].box_pos[1].x, x);
			sp_features[seg_id].box_pos[1].y = max(sp_features[seg_id].box_pos[1].y, y);

			// sum up boundary pixel number
			if(x==0 || x==cimg.cols-1 || y==0 || y==cimg.rows-1)
			{
				sp_features[seg_id].bnd_pixels++;
				sp_features[seg_id].perimeter++;	// boundary pixel must be in perimeter
				continue;
			}

			// perimeter
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y, x-1))
			{ sp_features[seg_id].perimeter++; continue; }
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y, x+1))
			{ sp_features[seg_id].perimeter++; continue; }
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y-1, x))
			{ sp_features[seg_id].perimeter++; continue; }
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y+1, x))
			{ sp_features[seg_id].perimeter++; continue; }

			if(SegSuperPixelFeature::use4Neighbor) continue;
			// 8 neighbor case
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y-1, x-1))
			{ sp_features[seg_id].perimeter++; continue; }
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y-1, x+1))
			{ sp_features[seg_id].perimeter++; continue; }
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y+1, x-1))
			{ sp_features[seg_id].perimeter++; continue; }
			if(seg_id != imgSegmentor.m_idxImg.at<int>(y+1, x+1))
			{ sp_features[seg_id].perimeter++; continue; }

		}
	}

	const int quantBins[3] = {4, 8, 8}; //used in paper: L_A_B
	int depth_bin_num = 10;
	float depth_bin_step = 255.f / depth_bin_num;
	for(size_t i=0; i<sp_features.size(); i++)
	{
		sp_features[i].id = i;
		//  init
		if( (saltype & SAL_COLOR) != 0 )
			sp_features[i].feat.resize(quantBins[0]+quantBins[1]+quantBins[2], 0);
		if( (saltype & SAL_DEPTH) != 0 )
			sp_features[i].dfeat.resize(depth_bin_num, 0);
		//	set bound box
		sp_features[i].box = \
			Rect(sp_features[i].box_pos[0].x, sp_features[i].box_pos[0].y, \
			sp_features[i].box_pos[1].x-sp_features[i].box_pos[0].x+1, sp_features[i].box_pos[1].y-sp_features[i].box_pos[0].y+1);
		sp_features[i].centroid.x /= sp_features[i].area;
		sp_features[i].centroid.y /= sp_features[i].area;
	}

	// compute segment feature: color and depth
	for(int y=0; y<cimg.rows; y++)
	{
		for(int x=0; x<cimg.cols; x++)
		{
			if( (saltype & SAL_DEPTH) != 0 )
			{
				/**** depth ****/
				// get rgb pixel from image
				float curd = dmap.at<float>(y, x);
				int curid = MIN(curd/depth_bin_step, depth_bin_num-1);

				sp_features[imgSegmentor.m_idxImg.at<int>(y, x)].dfeat[curid]++;
			}

			if( (saltype & SAL_COLOR) != 0 )
			{
				/**** color ****/
				// get rgb pixel from image
				Vec3b pixels = cimg.at<Vec3b>(y,x);
				// B_G_R
				ColorConvertHelper::LAB v = ColorConvertHelper::RGBtoLAB(pixels[2], pixels[1], pixels[0]);

				float l = v.L;
				float a = v.A;	a += 120;
				float b = v.B;	b += 120;
				int lbin = (int)(l/(100.f/quantBins[0]));
				lbin = ( lbin > quantBins[0]-1? quantBins[0]-1: lbin );
				int abin = (int)(a/(240.f/quantBins[1]));
				abin = ( abin > quantBins[1]-1? quantBins[1]-1: abin );
				int bbin = (int)(b/(240.f/quantBins[2]));
				bbin = ( bbin > quantBins[2]-1? quantBins[2]-1: bbin );

				sp_features[imgSegmentor.m_idxImg.at<int>(y, x)].feat[lbin]++;
				sp_features[imgSegmentor.m_idxImg.at<int>(y, x)].feat[quantBins[0]+abin]++;
				sp_features[imgSegmentor.m_idxImg.at<int>(y, x)].feat[quantBins[0]+quantBins[1]+bbin]++;
			}
		}
	}
	//do feature normalization
	for(size_t i=0; i<sp_features.size(); i++)
	{
		SegSuperPixelFeature& curfeat = sp_features[i];
		for(size_t j=0; j<curfeat.feat.size(); j++)
			curfeat.feat[j] /= (3*curfeat.area);
		for(size_t j=0; j<curfeat.dfeat.size(); j++)
			curfeat.dfeat[j] /= curfeat.area;
	}

	// init composer
	use_feat = saltype;
	if (!Base::Init(imgSegmentor.m_idxImg, sp_features)) return false;


	return true;
}

bool SalientRGBDRegionDetector::RankWins(vector<ImgWin>& wins)
{
	// compute composition cost for each window
	for (size_t i=0; i<wins.size(); i++)
	{
		wins[i].score = Compose(wins[i]);
	}

	sort(wins.begin(), wins.end(), [](const ImgWin& a, const ImgWin& b) { return a.score > b.score; } );

	return true;
}