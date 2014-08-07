
#include "GenericObjectDetector.h"



GenericObjectDetector::GenericObjectDetector(void)
{
	shiftCrit.maxCount = 10;
	shiftCrit.epsilon = 0.0001f;

	// detection window size
	winconfs.push_back(WinConfig(100, 200));
	//winconfs.push_back(WinConfig(200, 200));
	winconfs.push_back(WinConfig(200, 300));
	winconfs.push_back(WinConfig(300, 200));

	//
	bingObjectness = NULL;
	isBingInitialized = false;
}

GenericObjectDetector::~GenericObjectDetector()
{
	if( bingObjectness != NULL )
	{
		delete bingObjectness;
		bingObjectness = NULL;
	}
}

//////////////////////////////////////////////////////////////////////////

bool GenericObjectDetector::CreateScoremapFromWins(int imgw, int imgh, const vector<ImgWin>& imgwins, Mat& scoremap)
{
	scoremap.create(imgh, imgw, CV_32F);
	scoremap.setTo(0);

	for(size_t i=0; i<imgwins.size(); i++)
	{
		scoremap(imgwins[i]) += imgwins[i].score;
	}

	scoremap /= imgwins.size();

	return true;
}

//////////////////////////////////////////////////////////////////////////

bool GenericObjectDetector::Preprocess(const cv::Mat& color_img)
{
	imgSize.width = color_img.cols;
	imgSize.height = color_img.rows;

	// convert to gray
	//Mat grayimg;
	//cv::cvtColor(color_img, grayimg, CV_BGR2GRAY);

	//// compute gradient
	//Sobel(grayimg, Gx, CV_32F, 1, 0, 3);
	//double maxv, minv;
	//minMaxLoc(Gx, &minv, &maxv);
	//Sobel(grayimg, Gy, CV_32F, 0, 1, 3);
	///*ImgVisualizer::DrawFloatImg("gx", Gx, Mat());
	//ImgVisualizer::DrawFloatImg("gy", Gy, Mat());*/

	//// magnitude
	//magnitude(Gx, Gy, Gmag);
	//ImgVisualizer::DrawFloatImg("gmag", Gmag, Mat());

	//phase(Gx, Gy, Gdir, true);

	//// compute integrals for gx, gy
	//cv::integral(Gx, integralGx, CV_64F);
	//cv::integral(Gy, integralGy, CV_64F);

	// compute color integrals
	cv::Mat labImg;
	cv::cvtColor(color_img, labImg, CV_BGR2Lab);

	// split channels
	std::vector<cv::Mat> colorChannels(3);
	cv::split(labImg, colorChannels);

	// compute integrals
	colorIntegrals.resize(3);
	for(int i=0; i<3; i++) cv::integral(colorChannels[i], colorIntegrals[i], CV_64F);

	// prepare data
	/*FileInfos dmaps;
	db_man.GetDepthmapList(dmaps);
	db_man.LoadDepthData(dmaps[0].filepath, depthMap);*/

	return true;
}

double GenericObjectDetector::ComputeObjectScore(Rect win)
{
	// compute edge orientation difference sum
	double sumgx = integralGx.at<double>(win.br().y, win.br().x);
	sumgx += integralGx.at<double>(win.tl().y, win.tl().x);
	sumgx -= integralGx.at<double>(win.tl().y, win.br().x);
	sumgx -= integralGx.at<double>(win.br().y, win.tl().x);

	double sumgy = integralGy.at<double>(win.br().y, win.br().x);
	sumgy += integralGy.at<double>(win.tl().y, win.tl().x);
	sumgy -= integralGy.at<double>(win.tl().y, win.br().x);
	sumgy -= integralGy.at<double>(win.br().y, win.tl().x);

	return 1 / sqrt(sumgx*sumgx + sumgy*sumgy);
}

double GenericObjectDetector::ComputeDepthVariance(Rect win)
{
	cv::Scalar mean, stddev;
	meanStdDev(depthMap(win), mean, stddev);
	return 1.f / (stddev.val[0]+0.000005f);
}

double GenericObjectDetector::ComputeCenterSurroundMeanColorDiff(ImgWin win)
{
	// context window
	ImgWin contextWin = ToolFactory::GetContextWin(imgSize.width, imgSize.height, win, 2);

	// center mean
	Scalar centerMean, contextMean;
	for (int i=0; i<3; i++) 
	{
		centerMean.val[i] = ToolFactory::GetIntegralValue(colorIntegrals[i], win);
		contextMean.val[i] = ToolFactory::GetIntegralValue(colorIntegrals[i], contextWin);
		contextMean.val[i] -= centerMean.val[i];

		// average
		centerMean.val[i] /= win.area();
		contextMean.val[i] /= (contextWin.area() - win.area());
	}

	double score = 0;
	for(int i=0; i<3; i++) score += (centerMean.val[i]-contextMean.val[i])*(centerMean.val[i]-contextMean.val[i]);
	score = sqrt(score);

	return score;

}

//////////////////////////////////////////////////////////////////////////

bool GenericObjectDetector::WinLocRange(const Rect spbox, const WinConfig winconf, Point& minPt, Point& maxPt)
{
	if(spbox.width >= winconf.width || spbox.height >= winconf.height)
	{
		cerr<<"Detection window is smaller than superpixel box."<<endl;
		return false;
	}

	minPt.x = spbox.br().x - winconf.width;
	minPt.y = spbox.br().y - winconf.height;
	minPt.x = MAX(minPt.x, 0);
	minPt.y = MAX(minPt.y, 0);
	maxPt.x = spbox.x;
	maxPt.y = spbox.y;

	assert(minPt.x <= maxPt.x && minPt.y <= maxPt.y);

	return true;
}

bool GenericObjectDetector::SampleWinLocs(const Point startPt, const WinConfig winconf, const Point minPt, const Point maxPt, int num, vector<ImgWin>& wins)
{
	/*if(startPt.x < minPt.x || startPt.x > maxPt.x || startPt.y < minPt.y || startPt.y > maxPt.y)
	{
	cerr<<"start point is not in the valid range."<<endl;
	return false;
	}*/

	// x range; y range
	Point startTLPt(startPt.x - winconf.width/2, startPt.y - winconf.height/2);
	if(startTLPt.x < 0 || startTLPt.y < 0)
		return false;

	// restrict to move in a bounding box
	double minx = startTLPt.x - minPt.x;
	double maxx = maxPt.x - startTLPt.x;
	double xrange = MIN(minx, maxx);
	double miny = startTLPt.y - minPt.y;
	double maxy = maxPt.y - startTLPt.y;
	double yrange = MIN(miny, maxy);

	cv::RNG rng;
	// generate randomly point within the range
	wins.resize(num);
	for(size_t i=0; i<num; i++)
	{
		// select a new center
		double xdiff = rng.uniform(-xrange, xrange);
		double ydiff = rng.uniform(-yrange, yrange);
		wins[i].x = startTLPt.x + xdiff;
		wins[i].y = startTLPt.y + ydiff;
		if(wins[i].x + winconf.width >= imgSize.width || wins[i].y + winconf.height >= imgSize.height)
		{
			// remove invalid window
			i--;
			continue;
		}
		wins[i].width = winconf.width;
		wins[i].height = winconf.height;
	}

	//cout<<"Sampled window locations."<<endl;

	return true;
}


//////////////////////////////////////////////////////////////////////////

bool GenericObjectDetector::test()
{
	// test oversegmentation
	// get one image and depth map
	FileInfos imglist;
	db_man.Init(DB_VOC07);
	if( !db_man.GetImageList(imglist) )
		return false;
	map<string, vector<ImgWin>> gt_wins;
	db_man.LoadGTWins(imglist, gt_wins);
	img = imread(imglist[58].filepath);
	//db_man.GetDepthmapList(imglist);
	//db_man.LoadDepthData(imglist[0].filepath, depthMap);

	vector<ImgWin> wins;
	a9win.GenerateBlocks(img, wins);
	visualsearch::ImgVisualizer::DrawImgWins("detection", img, wins);

	WindowEvaluator wineval;
	vector<ImgWin> bestWins;
	wineval.FindBestWins(wins, gt_wins[imglist[58].filename], bestWins);
	visualsearch::ImgVisualizer::DrawImgWins("matches", img, bestWins);

	return true;

	// preprocessing
	Preprocess(img);

	segmentor.DoSegmentation(img);
	const vector<visualsearch::SuperPixel>& sps = segmentor.superPixels;

	int cnt = 10;
	while(cnt >= 0)
	{
		// select a segment
		srand(time(NULL));
		int sel_id = rand()%sps.size();
		vector<ImgWin> imgwins;
		// place window at center
		ImgWin curwin;
		curwin.x = sps[sel_id].box.x+sps[sel_id].box.width/2 - winconfs[0].width/2;
		curwin.y = sps[sel_id].box.y+sps[sel_id].box.height/2 - winconfs[0].height/2;
		curwin.width = winconfs[0].width;
		curwin.height = winconfs[0].height;
		imgwins.push_back(curwin);
		curwin = ImgWin(sps[sel_id].box.x, sps[sel_id].box.y, sps[sel_id].box.width, sps[sel_id].box.height);
		imgwins.push_back(curwin);
		visualsearch::ImgVisualizer::DrawImgWins("img", img, imgwins);
		visualsearch::ImgVisualizer::DrawImgWins("seg", segmentor.m_segImg, imgwins);
		visualsearch::ImgVisualizer::DrawImgWins("meancolor", segmentor.m_mean_img, imgwins);

		// compute adjust range
		Point minpt, maxpt;
		if( !WinLocRange(sps[sel_id].box, winconfs[0], minpt, maxpt) )
			continue;

		// check range
		//ImgWin minWin(minpt.x-winconfs[0].width/2, minpt.y-winconfs[0].height/2, winconfs[])

		Point curpt(sps[sel_id].box.x+sps[sel_id].box.width/2, sps[sel_id].box.y+sps[sel_id].box.height/2);
		double bestscore = 0;
		ImgWin bestWin;
		// do shifting
		for(int i=0; i<10; i++)
		{
			// generate locations
			vector<ImgWin> wins;
			SampleWinLocs(curpt, winconfs[0], minpt, maxpt, 6, wins);
			if(wins.empty())
				continue;

			for(size_t j=0; j<wins.size(); j++)
				wins[j].score = ComputeCenterSurroundMeanColorDiff(wins[j]);

			// sort
			sort(wins.begin(), wins.end());
			
			const ImgWin selWin = wins[wins.size()-1];
			cout<<"Best score: "<<selWin.score<<endl;
			if(selWin.score > bestscore)
			{
				// shift to max point
				curpt.x = selWin.x + selWin.width/2;
				curpt.y = selWin.y + selWin.height/2;

				// update
				bestWin = selWin;

				// visualize
				vector<ImgWin> imgwins2;
				ImgWin spwin = ImgWin(sps[sel_id].box.x, sps[sel_id].box.y, sps[sel_id].box.width, sps[sel_id].box.height);
				imgwins2.push_back(spwin);
				imgwins2.push_back(bestWin);
				visualsearch::ImgVisualizer::DrawImgWins("shift", img, imgwins2);
				cv::waitKey(0);
				cv::destroyWindow("shift");

				cout<<"Updated best score."<<endl;
			}

		}

		cnt--;
	}

	return true;
}

bool GenericObjectDetector::ShiftWindow(const Point& seedPt, Size winSz, Point& newPt)
{
	// define window
	Point topleft(seedPt.x-winSz.width/2, seedPt.y-winSz.height/2);
	topleft.x = MAX(0, topleft.x);
	topleft.y = MAX(0, topleft.y);

	Point bottomright(seedPt.x+winSz.width/2, seedPt.y+winSz.height/2);
	bottomright.x = MIN(bottomright.x, imgSize.width-1);
	bottomright.y = MIN(bottomright.y, imgSize.height-1);

	Rect win(topleft, bottomright);

	// start iteration
	for(int i=0; i<shiftCrit.maxCount; i++)
	{
		// compute edge orientation difference sum
		double sumgx = integralGx.at<double>(win.br().y, win.br().x);
		sumgx += integralGx.at<double>(win.tl().y, win.tl().x);
		sumgx -= integralGx.at<double>(win.tl().y, win.br().x);
		sumgx -= integralGx.at<double>(win.br().y, win.tl().x);

		double sumgy = integralGy.at<double>(win.br().y, win.br().x);
		sumgy += integralGy.at<double>(win.tl().y, win.tl().x);
		sumgy -= integralGy.at<double>(win.tl().y, win.br().x);
		sumgy -= integralGy.at<double>(win.br().y, win.tl().x);


	}

	return true;
}

bool GenericObjectDetector::ShiftWindowToMaxScore(const Point& seedPt, Point& newPt)
{
	// sample points


	// compute score

	// shift to max point

	return true;
}

bool GenericObjectDetector::RunSlidingWin(const cv::Mat& color_img, Size winsz)
{
	if(winsz.width > imgSize.width || winsz.height > imgSize.height)
		return false;

	cout<<"Running sliding window detection..."<<endl;

	cv::Mat scoremap(imgSize.height, imgSize.width, CV_64F);
	scoremap.setTo(0);

	vector<ImgWin> wins;

	for(int r=0; r<imgSize.height-winsz.height-1; r++)
	{
		for(int c=0; c<imgSize.width-winsz.width-1; c++)
		{
			ImgWin curwin(c, r, winsz.width, winsz.height);
			curwin.score = ComputeDepthVariance(curwin); //ComputeObjectScore(curwin);
			scoremap.at<double>(r+winsz.height/2, c+winsz.width/2) = curwin.score;

			wins.push_back(curwin);
		}
	}

	sort(wins.begin(), wins.end());

	visualsearch::ImgVisualizer::DrawFloatImg("scoremap", scoremap, Mat());
	visualsearch::ImgVisualizer::DrawImgWins("img", color_img, wins);
	waitKey(10);

	return true;
}

bool GenericObjectDetector::Run(const cv::Mat& color_img, vector<ImgWin>& det_wins)
{
	if( !Preprocess(color_img) )
		return false;


	return true;
}

bool GenericObjectDetector::RunVOC()
{

	FileInfos imglist;
	db_man.Init(DB_VOC07);
	if( !db_man.GetImageList(imglist) )
		return false;

	for(size_t id=0; id<10; id++)
	{
		double start_t = getTickCount();

		img = imread(imglist[id].filepath);

		// preprocessing
		Preprocess(img);

		segmentor.DoSegmentation(img);
		const vector<visualsearch::SuperPixel>& sps = segmentor.superPixels;

		vector<ImgWin> det_wins;

		// loop all segment
		for (size_t sel_id=0; sel_id<sps.size(); sel_id++)
		{
			// test all window settings
			for(size_t win_id=0; win_id<winconfs.size(); win_id++)
			{
				// compute adjust range
				Point minpt, maxpt;
				if( !WinLocRange(sps[sel_id].box, winconfs[win_id], minpt, maxpt) )
					continue;

				Point curpt(sps[sel_id].box.x+sps[sel_id].box.width/2, sps[sel_id].box.y+sps[sel_id].box.height/2);
				double bestscore = 0;
				ImgWin bestWin;
				// do shifting
				for(int i=0; i<10; i++)
				{
					// generate locations
					vector<ImgWin> wins;
					SampleWinLocs(curpt, winconfs[win_id], minpt, maxpt, 6, wins);
					if(wins.empty())
						continue;

					for(size_t j=0; j<wins.size(); j++)
						wins[j].score = ComputeCenterSurroundMeanColorDiff(wins[j]);

					// sort
					sort(wins.begin(), wins.end());

					const ImgWin selWin = wins[wins.size()-1];
					//cout<<"Best score: "<<selWin.score<<endl;
					if(selWin.score > bestscore)
					{
						// shift to max point
						curpt.x = selWin.x + selWin.width/2;
						curpt.y = selWin.y + selWin.height/2;

						// update
						bestWin = selWin;
					}
				}

				det_wins.push_back(bestWin);
			}
		}

		sort(det_wins.begin(), det_wins.end());

		cout<<"Time for image "<<id<<": "<<(getTickCount()-start_t) / getTickFrequency()<<"s"<<endl;

		// visualize final results
		reverse(det_wins.begin(), det_wins.end());

		visualsearch::ImgVisualizer::DrawImgWins("final", img, det_wins);

		waitKey(0);
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////

bool GenericObjectDetector::InitBingObjectness()
{
	// set dataset for training and testing
	DataSetVOC voc2007("e:\\VOC2007\\");
	//voc2007.loadAnnotations();

	// params
	double base = 2;
	int W = 8;
	int NSS = 2;
	int numPerSz = 130;

	bingObjectness = new Objectness(voc2007, base, W, NSS);
	if( bingObjectness->loadTrainedModel() != 1 )
	{
		cerr<<"Fail to load all bing model."<<endl;
		return false;
	}

	isBingInitialized = true;

	return true;
}

bool GenericObjectDetector::GetObjectsFromBing(const cv::Mat& cimg, vector<ImgWin>& detWins, int winnum, bool showres)
{
	if( !isBingInitialized )
	{
		cerr<<"Bing is not initialized."<<endl;
		return false;
	}

	ValStructVec<float, Vec4i> boxes;
	bingObjectness->getObjBndBoxes(cimg, boxes);

	int validwinnum = MIN(winnum, boxes.size());
	detWins.clear();
	detWins.resize(validwinnum);
	for (int i=0; i<validwinnum; i++)
	{
		detWins[i] = ImgWin( boxes[i].val[0], boxes[i].val[1], boxes[i].val[2]-boxes[i].val[0], boxes[i].val[3]-boxes[i].val[1] );
		detWins[i].score = 1;
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////

bool GenericObjectDetector::ProposeObjects(const Mat& cimg, const Mat& dmap, vector<ImgWin>& objwins, vector<ImgWin>& salwins, bool ifRank)
{
	if( !isBingInitialized )
		if( !InitBingObjectness() )
			return false;

	// get objectness windows
	vector<ImgWin> objboxes;
	if( !GetObjectsFromBing(cimg, objboxes, 800) )
		return false;
	//visualsearch::ImgVisualizer::DrawImgWins("objectness", curimg, objboxes);

	// rank
	SalientRGBDRegionDetector saldet;
	vector<ImgWin> salboxes = objboxes;
	//depth_sal.RankWins(curdmap, salboxes);
	//saldet.g_para.segThresholdK = 200;
	if( ifRank )
	{
		saldet.Init(SAL_COLOR, cimg, dmap);
		saldet.RankWins(salboxes);
	}

	// only select windows containing center point and not having a dimension bigger than half
	objwins.clear();
	salwins.clear();
	Point centerp(cimg.cols/2, cimg.rows/2);
	for(size_t i=0; i<salboxes.size(); i++)
	{
		if(salboxes[i].contains(centerp) && (salboxes[i].width < cimg.cols*2/3 || salboxes[i].height < cimg.rows*2/3))
			salwins.push_back(salboxes[i]);
	}
	for(size_t i=0; i<objboxes.size(); i++)
	{
		if(objboxes[i].contains(centerp) && (objboxes[i].width < cimg.cols*2/3 || objboxes[i].height < cimg.rows*2/3))
			objwins.push_back(objboxes[i]);
	}

	return true;
}