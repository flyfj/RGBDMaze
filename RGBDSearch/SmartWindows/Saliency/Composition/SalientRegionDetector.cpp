#include "SalientRegionDetector.h"

//#include <BitmapHelper.h>
#include <algorithm>
#include <fstream>
#include "colorconverthelper.h"
#include "nms.h"


wstring string2wstring(string str)  
{  
	wstring result;   
	int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), NULL, 0);  
	LPWSTR buffer = new wchar_t[len + 1];  
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), buffer, len);  
	buffer[len] = '\0';  
	result.append(buffer);
	delete[] buffer;
	return result;
}

inline void fast_2d_nms(const ImageFloatSimple& score_map, int winw, int winh, float nms_threshold, vector<ScoredRect>& wins)
{	
	// TODO: briefly describe the method and add comments to each block
	if(!wins.empty())
		wins.clear();

	assert((nms_threshold > 0) && (nms_threshold < 1));
	const int imgw = score_map.Width();
	const int imgh = score_map.Height();

	// TODO: what's the meaning
	//int ystep = (int)(2*winh*nms_threshold/(nms_threshold+1));
	//int xstep = (int)(2*winw*nms_threshold/(nms_threshold+1));

	int ystep = winh*(1-nms_threshold)/(nms_threshold+1) + 0.5;
	int xstep = winw*(1-nms_threshold)/(nms_threshold+1) + 0.5;

	ScoredRect win(Rect(0, 0, winw, winh));
	
	for (int y = 0; y < imgh-winh+1; y += ystep) 
	for (int x = 0; x < imgw-winw+1; x += xstep)
	{
		// find maximum in each block
		win.score = score_map.Pixel(x, y);
		// ? why not (x,y)
		win.x = x-winw/2;
		win.y = y-winh/2;
		//win.X = x; win.Y = y;
		for (int yy = y; yy < y+winh; yy++)	
		for (int xx = x; xx < x+winw; xx++)
		{
			if (score_map.Pixel(xx, yy) > win.score)
			{
				win.score = score_map.Pixel(xx, yy);
				win.x = xx-winw/2;
				win.y = yy-winh/2;
			}
		}

		if(win.score == 0)
			continue;

		wins.push_back(win);
	}	
}


///////////////////////////////////////////////////////////////////////////////
SalientRegionDetector::SalientRegionDetector() : m_nSlideStep(0)
{
	// set detection params
	g_win_para = SlideWinPara(0.02f, 0.5f, 6, 0.5f, 2.0f, 7);
}

SalientRegionDetector::~SalientRegionDetector()
{
	Clear();
}

void SalientRegionDetector::Clear()
{
	Base::Clear();

	segmentedImg.clear();
	sp_features.clear();
	nms_vals.clear();
	all_objs.clear();

}

bool SalientRegionDetector::Init(const Mat& img)
{
	// clear data for new image
	Clear();

	// set multi-nms values
	{
		if( g_para.nms_step == 0 )
			nms_vals.push_back( g_para.nms_min );
		else
		{
			for( float fv = g_para.nms_min; fv <= g_para.nms_max; fv += g_para.nms_step )
				nms_vals.push_back(fv);
		}
		
		// init object container
		all_objs.resize(nms_vals.size());
	}

	segmentedImg.resize(3*img.cols*img.rows);

	// run segmentation
	float sigma = g_para.segSigma, c = g_para.segThresholdK; 
	int min_size = g_para.segMinArea;

	Mat segmentMat;
	imgSegmentor.m_dMinArea = min_size;
	imgSegmentor.m_dSmoothSigma = sigma;
	int superpixel_num = imgSegmentor.DoSegmentation(img);
		//graph_based_segment(img, sigma, c, min_size, imgSegmentor.m_idxImg, &segmentedImg[0]);
	g_runinfo.seg_num = superpixel_num;
	// visualize segment image
	//imshow("segmentimage", imgSegmentor.m_segImg);
	//waitKey(0);

	//////////////////////////////////////////////////////////////////////////
	// compute features for each superpixel
	//////////////////////////////////////////////////////////////////////////
	// create data
	sp_features.resize(superpixel_num);

	// compute perimeter and boundary pixel numbers, set bounding box
	for(int y=0; y<img.rows; y++)
	{
		for(int x=0; x<img.cols; x++)
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
			if(x==0 || x==img.cols-1 || y==0 || y==img.rows-1)
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
	for(size_t i=0; i<sp_features.size(); i++)
	{
		sp_features[i].id = i;
		//  normalize
		sp_features[i].feat.resize(quantBins[0]+quantBins[1]+quantBins[2], 0);
		//	set bound box
		sp_features[i].box = \
			Rect(sp_features[i].box_pos[0].x, sp_features[i].box_pos[0].y, \
			sp_features[i].box_pos[1].x-sp_features[i].box_pos[0].x+1, sp_features[i].box_pos[1].y-sp_features[i].box_pos[0].y+1);
		sp_features[i].centroid.x /= sp_features[i].area;
		sp_features[i].centroid.y /= sp_features[i].area;
	}

	// compute appearance feature: LAB histogram
	for(int y=0; y<img.rows; y++)
	{
		for(int x=0; x<img.cols; x++)
		{
			// get rgb pixel from image
			Vec3b pixels = img.at<Vec3b>(y,x);
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
	//do feature normalization
	for(size_t i=0; i<sp_features.size(); i++)
	{
		SegSuperPixelFeature& curfeat = sp_features[i];
		for(size_t j=0; j<curfeat.feat.size(); j++)
				curfeat.feat[j] /= (3*curfeat.area);
	}

	// compute background weight map
	/*if(g_para.useBGMap)
	ComputeBGMap(img);*/

	// init composer
	if (!Base::Init(imgSegmentor.m_idxImg, sp_features)) return false;

	return true;
}

void SalientRegionDetector::RunSlidingWindow(const int win_width, const int win_height)
{
	if (g_para.slidingStep > 1)	// brute force
	{
		vector<ScoredRect> wins;
		wins.reserve((m_nImgHeight - win_height+1) * (m_nImgWidth - win_width+1));

		ScoredRect win(Rect(0, 0, win_width, win_height));
		for(win.y = 0; win.y < m_nImgHeight - win.height+1; win.y += g_para.slidingStep)
		{
			for(win.x = 0; win.x < m_nImgWidth - win.width+1; win.x += g_para.slidingStep)
			{				
				win.score = Compose(win);
				wins.push_back(win);	// brute force: record every window
				compose_cost_map.Pixel(win.x+win.width/2, win.y+win.height/2) = win.score;
			}
		}
	}
	else
	{
		ComposeAll(win_width, win_height, g_para.slidingStep == 0);
	}

	// do fast nms first for scoremap
	for(size_t i=0; i<nms_vals.size(); i++)
		fast_2d_nms(compose_cost_map, win_width, win_height, nms_vals[i], all_objs[i]);

	// do nms for all local nms windows from fast nms
	// get final results for current scale
	for(size_t i=0; i<nms_vals.size(); i++)
		all_objs[i] = nms(all_objs[i], nms_vals[i]);

}

int SalientRegionDetector::RunMultiSlidingWindow()
{
	const int img_width = m_nImgWidth;
	const int img_height = m_nImgHeight;
	const int imgArea = m_nImgWidth * m_nImgHeight;

	// optimal windows for each nms value
	vector<vector<ScoredRect>> temp_optimals;	
	temp_optimals.resize(nms_vals.size());
	for(size_t i=0; i<temp_optimals.size(); i++)
		temp_optimals[i].reserve(m_nImgWidth*m_nImgHeight/8);	// speed up push_back

	int win_size_count = 0;
	// loop every sliding window scale
	#pragma omp parallel for
	for(int step = 0; step < g_win_para.areas.size(); step++)
	{
		const float curArea = imgArea * g_win_para.areas[step];
		for(int idx = 0; idx < g_win_para.aspect_ratios.size(); idx++)
		{
			const int win_width = (int)( sqrt(curArea / g_win_para.aspect_ratios[idx]) );
			const int win_height = (int)( sqrt(curArea * g_win_para.aspect_ratios[idx]) );
			if ((win_width > img_width) || (win_height > img_height))	continue;

			win_size_count++;

			{
				// run each window scale
				RunSlidingWindow(win_width, win_height);
				printf("*");
			}

			// accumulate local optimal windows for different nms value
			for (size_t i = 0; i < all_objs.size(); i++)
			{
				// different windows
				for(size_t j = 0; j < all_objs[i].size(); j++)
					if( all_objs[i][j].score > g_para.saliencyThre )	// filter non-valid window
						temp_optimals[i].push_back(all_objs[i][j]);
			}
		}
	}

	// deal with different nms value
	for(size_t i=0; i<nms_vals.size(); i++)
		all_objs[i] = nms(temp_optimals[i], nms_vals[i]);

	return win_size_count;

}

//////////////////////////////////////////////////////////////////////////
// added
//////////////////////////////////////////////////////////////////////////
bool SalientRegionDetector::RankWins(vector<ImgWin>& wins)
{
	// compute composition cost for each window
	for (size_t i=0; i<wins.size(); i++)
	{
		wins[i].score = Compose(wins[i]);
	}

	sort(wins.begin(), wins.end(), [](const ImgWin& a, const ImgWin& b) { return a.score > b.score; } );

	return true;
}


void SalientRegionDetector::DrawResult(Mat& img, double down_ratio, const vector<ScoredRect>& objs) const
{
	char str[30];
	const int fontSize = (int)(9 / down_ratio);
	// save for single nms or multi_nms
	for (int i = min(g_para.bestN_drawn, (int)objs.size())-1; i >= 0; i--)
	{
		const ScoredRect& rc = objs[i];
		if (rc.score < g_para.saliencyThre)	continue;

		rectangle(img, rc.tl(), rc.br(), CV_RGB(255,255,0), 1.2f/down_ratio);

		sprintf(str, "%d", i+1);
		string id(str);
		Size textsize = getTextSize(id, CV_FONT_HERSHEY_SIMPLEX, fontSize, 1, 0);
		putText(img, id, Point(rc.x+30, rc.y+30), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255,0,0));
	}
}

//bool SalientRegionDetector::SaveSegmentImage(const WCHAR* filename) const
//{
//	/*Bitmap bmp(m_nImgWidth, m_nImgHeight, 3*m_nImgWidth, PixelFormat24bppRGB, const_cast<BYTE*>(&segmentedImg[0]));
//
//	return SaveBitmap(bmp, L"image/jpeg", filename);*/
//
//	return true;
//}

void SalientRegionDetector::SaveDetectionResults(string save_img_prefix)
{
	for( size_t i=0; i<nms_vals.size(); i++ )
	{
		string res_file;
		char str[30];
		sprintf(str, "%.2f", nms_vals[i]);
		res_file = save_img_prefix + "_" + string(str) + "_res.txt";
		//g_runinfo.det_num = all_objs.size();

		ofstream out(res_file.c_str());
		if (!out)
			printf("fail to save text result in .txt\n");
		else
		{
			// output all detection results
			for (size_t j=0; j<all_objs[i].size(); j++)
			{
				const ScoredRect& rc = all_objs[i][j];
				out << rc.x << " " << rc.y << " " << rc.width << " " << rc.height << " " << rc.score << endl;
			}
		}
		out.close();
	}

}

void SalientRegionDetector::SaveBGMap(string save_img_prefix)
{
	//string res_file = save_img_prefix + "_bg.jpg";
	//// convert to bitmap (do normalization)
	//Bitmap bg_disp(bgMap.Width(), bgMap.Height());
	//for(int y=0; y<bg_disp.GetHeight(); y++)
	//{
	//	for(int x=0; x<bg_disp.GetWidth(); x++)
	//	{
	//		Color cur_color((BYTE)(bgMap.Pixel(x,y)*255), (BYTE)(bgMap.Pixel(x,y)*255), (BYTE)(bgMap.Pixel(x,y)*255));
	//		bg_disp.SetPixel(x, y, cur_color);
	//	}
	//}

	//if ( !SaveBitmap(bg_disp, L"image/jpeg", string2wstring(res_file).c_str()) )
	//	cout << "fail to save bg map in " << res_file << endl;
}
//////////////////////////////////////////////////////////////////////////
void SalientRegionDetector::print_help()
{
	printf("Invalid arguments: detector.exe image_folder output_folder [-d downsample_rate] [-b] [-k value] [-ss] [-tnms min max step] [-tscore value] [-o out_num] [-a min max level] [-r min max level].\n");
	printf("\ninput_folder (jpg|jpeg|bmp|gif|png)\n");
	printf("e.g. detector.exe d:\\testimgs d:\\res\n");
	printf("\n-d downsample_rate: if d<1, down sample image to percentage d; if image longer dimension > d, resize it so the longer dimension is d (default: 300)\n");
	printf("\n-b: enable usage of background probability map (default: disabled)\n");
	printf("\n-k value: segmentation parameter, the larger the fewer segments (default: 200)\n");
	printf("\n-ss: enable saving segment map (default: disabled)\n");
	printf("\n-tnms min max step: set the nms threshold range within [0, 1], if min equals max, only one value is used (default: 0.6, 0.6, 0)\n");
	printf("\n-tscore value: only save results with score > value (default: 0.08)\n");
	printf("\n-o out_num: the top out_num windows are displayed in the result (default: 5)\n");
	printf("\n-a min max level: set sliding window areas, default=(0.02 0.5 6) means min(max) win area is 2%(50%) of image area and there are 6 window areas\n");
	printf("\n-r min max level: set sliding window aspect ratio (height/width), default=(0.5 2.0 7) means min(max) ratio is 0.5(2.0) and there are 7 ratios\n");
}

bool SalientRegionDetector::read_para(int argc, _TCHAR* argv[])
{
	// reset params
	g_para = DetectionParams();

	// first check argument number
	int rightNum = 3;
	for(int i=0; i<argc; i++)
	{
		string nextParams( argv[i] );
		if (nextParams == "-d")
			rightNum += 2;
		else if (nextParams == "-b")
			rightNum += 1;
		else if (nextParams == "-k")
			rightNum += 2;
		else if (nextParams == "-ss")
			rightNum += 1;
		else if (nextParams == "-tnms")
			rightNum += 4;
		else if (nextParams == "-tscore")
			rightNum += 2;
		else if (nextParams == "-o")
			rightNum += 2;
		else if (nextParams == "-a")
			rightNum += 4;
		else if (nextParams == "-r")
			rightNum += 4;
	}

	if (rightNum != argc)
	{
		print_help();		
		return false;
	}

	// parse command line parameters
	g_para.g_imagefile = argv[1];
	g_para.g_outfile = argv[2];
	int argid = 3;
	while(argid < argc)
	{
		string nextParams( argv[argid++] );
		if ( nextParams == "-d" )
		{
			g_para.downSampleFactor = atof( argv[argid++] );
			if (g_para.downSampleFactor == 0)
			{
				printf("Wrong downsample factor.\n");
				return false;
			}
		}
		else if( nextParams == "-b" )
		{
			g_para.useBGMap = 1;
		}
		else if ( nextParams == "-k" )
		{
			g_para.segThresholdK = atof( argv[argid++] );
			if (g_para.segThresholdK == 0)
			{
				printf("Wrong segment threshold.\n");
				return false;
			}
		}
		else if( nextParams == "-ss" )
		{
			g_para.saveSegmap = 1;
		}
		else if(nextParams == "-tnms")
		{
			/*g_para.nmsTh = atof( argv[argid++] );
			if (g_para.nmsTh == 0)
				g_para.useMultiNMS = 1;
			else if(g_para.nmsTh < 0.4f || g_para.nmsTh > 0.8)
			{
				printf("Invalid nms value, should be within [0.4, 0.8].\n");
				return false;
			}*/

			g_para.nms_min = atof( argv[argid++] );
			g_para.nms_max = atof( argv[argid++] );
			g_para.nms_step = atof( argv[argid++] );
			if (g_para.nms_min >= 0.f && g_para.nms_max <= 1.0f && 
				g_para.nms_step >= 0.f && g_para.nms_step <= 1.0f &&
				g_para.nms_max - g_para.nms_step >= g_para.nms_min )
			{
				if(g_para.nms_min == g_para.nms_max)	// one nms value
				{
					g_para.useMultiNMS = 0;
					g_para.nmsTh = g_para.nms_min;
				}
				else
					g_para.useMultiNMS = 1;
			}
			else
			{
				printf("Invalid nms values.\n");
				return false;
			}
		}
		else if(nextParams == "-tscore")
		{
			g_para.saliencyThre = atof( argv[argid++] );
		}
		else if(nextParams == "-o")
		{
			g_para.bestN_drawn = atoi( argv[argid++] );
			if (g_para.bestN_drawn <=0)
			{
				printf("Wrong output number(<=0).\n");
				return false;
			}
		}
		else if(nextParams == "-a")
		{
			float minArea = atof( argv[argid++] );
			float maxArea = atof( argv[argid++] );
			int areaStepNum = atoi( argv[argid++] );
			if (minArea == 0 || maxArea == 0 || areaStepNum == 0)
			{
				printf("Wrong area parameters.\n");
				return false;
			}
			else g_win_para.SetArea(minArea, maxArea, areaStepNum);
		}
		else if(nextParams == "-r")
		{
			float minRatio = atof( argv[argid++] );
			float maxRatio = atof( argv[argid++] );
			int ratioStepNum = atoi( argv[argid++] );
			if (minRatio == 0 || maxRatio == 0 || ratioStepNum == 0)
			{
				printf("Wrong ratio parameters.\n");
				return false;
			}
			else g_win_para.SetAspectRatio(minRatio, maxRatio, ratioStepNum);
		}
		else
		{
			printf("Wrong parameter: %s.\n", nextParams.c_str());
			return false;
		}
	}

	return true;
}


void SalientRegionDetector::ConvertSegmentImage2Mat(Mat &segmentMat, int width, int height)
{
	if(width*height*3 != segmentedImg.size())
	{
		cerr<<"Segmentation image is not in correct format."<<endl;
		return;
	}

	segmentMat.create(height, width, CV_8UC3);
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			Vec3b val;
			val.val[0] = segmentedImg[y*width*3+3*x];
			val.val[1] = segmentedImg[y*width*3+3*x+1];
			val.val[2] = segmentedImg[y*width*3+3*x+2];
			segmentMat.at<Vec3b>(y,x) = val;
		}
	}

}