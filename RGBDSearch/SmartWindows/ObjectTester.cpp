#include "ObjectTester.h"


ObjectTester::ObjectTester(void)
{
}

//////////////////////////////////////////////////////////////////////////

void ObjectTester::TestObjectRanking(const DatasetName& dbname)
{
	// load dataset
	FileInfos img_fns;
	FileInfos dmap_fns;
	map<string, vector<ImgWin>> rawgtwins;

	NYUDepth2DataMan nyudata;
	Berkeley3DDataManager berkeleydata;

	if(dbname == DB_NYU2_RGBD)
	{
		nyudata.GetImageList(img_fns);
		nyudata.GetDepthmapList(dmap_fns);
		if(img_fns.size() != dmap_fns.size())
			return;
		nyudata.LoadGTWins(img_fns, rawgtwins);
	}
	if(dbname == DB_BERKELEY3D)
	{
		berkeleydata.GetImageList(img_fns);
		berkeleydata.GetDepthmapList(dmap_fns);
		if(img_fns.size() != dmap_fns.size())
			return;
		berkeleydata.LoadGTWins(img_fns, rawgtwins);
	}

	GenericObjectDetector detector;
	if( !detector.InitBingObjectness() )
		return;

	SalientRegionDetector saldet;
	SalientRGBDRegionDetector salrgbd;
	DepthSaliency depth_sal;
	vector<vector<ImgWin>> objdetwins(img_fns.size()), saldetwins(img_fns.size()), depthdetwins;
	vector<vector<ImgWin>> gtwins(img_fns.size());

#pragma omp parallel for
	for (int i=0; i<img_fns.size(); i++)
	{
		// read image
		Mat curimg = imread(img_fns[i].filepath);
		if(curimg.empty())
			continue;

		// read depth
		Mat curdmap;
		if(dbname == DB_NYU2_RGBD)
			if( !nyudata.LoadDepthData(dmap_fns[i].filepath, curdmap) )
				continue;
		if(dbname == DB_BERKELEY3D)
			if( !berkeleydata.LoadDepthData(dmap_fns[i].filepath, curdmap) )
				continue;

//#define VERBOSE
#ifdef VERBOSE
		// show gt
		visualsearch::ImgVisualizer::DrawImgWins("gt", curimg, rawgtwins[img_fns[i].filename]);
		visualsearch::ImgVisualizer::DrawFloatImg("dmap", curdmap, Mat());
#endif

		// normalize to image
		//normalize(curdmap, curdmap, 0, 255, NORM_MINMAX);
		//curdmap.convertTo(curdmap, CV_8U);
		//cvtColor(curdmap, curdmap, CV_GRAY2BGR);
		//imshow("depthimg", curdmap);
		//waitKey(10);

		//visualsearch::ImgVisualizer::DrawImgWins("b3d", curimg, rawgtwins[img_fns[i].filename]);
		//waitKey(0);

		// resize
		Size newsz;
		//ToolFactory::compute_downsample_ratio(Size(curimg.cols, curimg.rows), 300, newsz);
		//resize(curimg, curimg, newsz);

		double start_t = getTickCount();
		// get objectness windows
		vector<ImgWin> objboxes;
		detector.GetObjectsFromBing(curimg, objboxes, 1000);
		//visualsearch::ImgVisualizer::DrawImgWins("objectness", curimg, objboxes);
		cout<<"objectness: "<<(double)(getTickCount()-start_t) / getTickFrequency()<<endl;
		
		start_t = getTickCount();
		// rank
		normalize(curdmap, curdmap, 0, 255, NORM_MINMAX);
		vector<ImgWin> salboxes = objboxes;
		int saltype = SAL_COLOR | SAL_DEPTH;
		salrgbd.Init(saltype, curimg, curdmap);
		salrgbd.RankWins(salboxes);
		//depth_sal.RankWins(curdmap, salboxes);
		cout<<"Depth ranking: "<<(double)(getTickCount()-start_t) / getTickFrequency()<<endl;

#ifdef VERBOSE
		vector<Mat> imgs(50);
		for (int i=0; i<50; i++)
		{
			imgs[i] = curimg(salboxes[i]);
		}
		Mat dispimg;
		visualsearch::ImgVisualizer::DrawImgCollection("objectness", imgs, 50, 15, dispimg);
		imshow("objectness", dispimg);
		visualsearch::ImgVisualizer::DrawImgWins("saldet", curimg, salboxes);
		waitKey(0);
#endif
		/*saldet.g_para.segThresholdK = 200;
		saldet.Init(curdmap);
		saldet.RankWins(salboxes);*/
		//visualsearch::ImgVisualizer::DrawImgWins("sal", curimg, salboxes);
		//waitKey(0);

		// add to collection
		objdetwins[i] = objboxes;
		saldetwins[i] = salboxes;
		gtwins[i] = rawgtwins[img_fns[i].filename];
		
		cout<<"Finish detection on "<<i<<"/"<<img_fns.size()<<endl;
	}

	// evaluation
	WindowEvaluator eval;
	vector<Point2f> objprvals, salprvals, depthprvals;
	int topnum[] = {1, 5, 10, 50, 100, 200, 500, 800, 1000};
	for(int i=0; i<9; i++)
	{
		Point2f curpr = eval.ComputePR(objdetwins, gtwins, topnum[i]);
		objprvals.push_back(curpr);
		curpr = eval.ComputePR(saldetwins, gtwins, topnum[i]);
		salprvals.push_back(curpr);
	}
	
	// save to file
	ofstream out1("nyu_objpr.txt");
	for (size_t i=0; i<objprvals.size(); i++) out1<<objprvals[i].x<<" "<<objprvals[i].y<<endl;
	ofstream out2("nyu_rgbdpr.txt");
	for (size_t i=0; i<salprvals.size(); i++) out2<<salprvals[i].x<<" "<<salprvals[i].y<<endl;

	cout<<"Finish evaluation"<<endl;

}


void ObjectTester::RunVideoDemo()
{
	GenericObjectDetector detector;

	KinectDataMan kinectDM;

	if( !kinectDM.InitKinect() )
		return;

	bool doRank = true;
	// start fetching stream data
	while(1)
	{
		Mat cimg, dmap;
		kinectDM.GetColorDepth(cimg, dmap);

		// resize image
		Size newsz;
		ToolFactory::compute_downsample_ratio(Size(cimg.cols, cimg.rows), 300, newsz);
		resize(cimg, cimg, newsz);
		//resize(dmap, dmap, newsz);
		//normalize(dmap, dmap, 0, 255, NORM_MINMAX);

		// get objects
		vector<ImgWin> objwins, salwins;
		if( !detector.ProposeObjects(cimg, dmap, objwins, salwins, doRank) )
			continue;

		//////////////////////////////////////////////////////////////////////////
		// draw best k windows
		int topK = MIN(6, objwins.size());
		int objimgsz = newsz.height / topK;
		int canvas_h = newsz.height;
		int canvas_w = newsz.width + 10 + objimgsz*2;
		Mat canvas(canvas_h, canvas_w, CV_8UC3);
		canvas.setTo(Vec3b(0,0,0));
		// get top windows
		vector<Mat> detimgs(topK);
		vector<Mat> salimgs(topK);
		for (int i=0; i<topK; i++)
		{
			cimg(objwins[i]).copyTo(detimgs[i]);
			resize(detimgs[i], detimgs[i], Size(objimgsz, objimgsz));
			cimg(salwins[i]).copyTo(salimgs[i]);
			resize(salimgs[i], salimgs[i], Size(objimgsz, objimgsz));
		}

		// draw boxes on input
		Scalar objcolor(0, 255, 0);
		Scalar salcolor(0, 0, 255);
		for(int i=0; i<topK; i++)
		{
			rectangle(cimg, objwins[i], objcolor);
			rectangle(cimg, salwins[i], salcolor);
		}
		circle(cimg, Point(cimg.cols/2, cimg.rows/2), 2, CV_RGB(255,0,0));
		// copy input image
		cimg.copyTo(canvas(Rect(0, 0, cimg.cols, cimg.rows)));

		// copy subimg
		for (int i=0; i<detimgs.size(); i++)
		{
			Rect objbox(cimg.cols+10, i*objimgsz, objimgsz, objimgsz);
			detimgs[i].copyTo(canvas(objbox));
			Rect salbox(cimg.cols+10+objimgsz, i*objimgsz, objimgsz, objimgsz);
			salimgs[i].copyTo(canvas(salbox));
		}

		resize(canvas, canvas, Size(canvas.cols*2, canvas.rows*2));
		//if(doRank)
			//putText(canvas, "Use Ranking", Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
		imshow("object proposals", canvas);
		if( waitKey(10) == 'q' )
			break;
	}
}