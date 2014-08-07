#include "SegmentBasedWindowComposer.h"

#include <algorithm>

bool SegSuperPixelFeature::use4Neighbor = false;

//////////////////////////////////////////////////////////////////////////
// compute spatial distance
float Point2RectDistance(int x, int y, const Rect& box)
{
	if (box.contains(Point(x,y))) return 0;

	if(x<box.x)
	{
		if(y<box.y)
			return l2_dist(x, y, box.x, box.y);
		else if(y<box.br().y)
			return (box.x-x);
		else
			return l2_dist(x, y, box.x, box.br().y);
	}
	else if(x<box.br().x)
	{
		if(y<box.y)
			return (box.y-y);
		else if(y>box.br().y)
			return (y-box.br().y);
		else
			return 0;
	}
	else
	{
		if(y<box.y)
			return l2_dist(x, y, box.br().x, box.y);
		else if(y<box.br().y)
			return (x-box.br().x);
		else
			return l2_dist(x, y, box.br().x, box.br().y);
	}
}

float HausdorffDist(Rect a, Rect b)
{
	// compute approximate Hausdorff distance
	// a->b
	float mindist1 = 999999999.f;
	Rect box1 = a;
	Rect box2 = b;

	vector<Point> points(4);
	points[0] = Point(box1.x, box1.y);
	points[1] = Point(box1.x, box1.br().y);
	points[2] = Point(box1.br().x, box1.y);
	points[3] = Point(box1.br().x, box1.br().y);
	
	for(size_t id = 0; id < 4; id++)
	{
		float dist = Point2RectDistance(points[id].x, points[id].y, box2);
		if(dist < mindist1)
			mindist1 = dist;
	}

	// b->a
	float mindist2 = 999999999.9f;
	points[0] = Point(box2.x, box2.y);
	points[1] = Point(box2.x, box2.br().y);
	points[2] = Point(box2.br().x, box2.y);
	points[3] = Point(box2.br().x, box2.br().y);

	for(size_t id = 0; id < 4; id++)
	{
		float dist = Point2RectDistance(points[id].x, points[id].y, box1);
		if(dist < mindist2)
			mindist2 = dist;
	}

	return max(mindist1, mindist2);
}

/////////////////////////////////////////////////////////////////////////////////////////////
SegmentBasedWindowComposer::SegmentBasedWindowComposer() : m_nImgWidth(0), m_nImgHeight(0)
	,m_fMaxCost(0.7f)
{
	use_feat = SAL_COLOR;
}

SegmentBasedWindowComposer::~SegmentBasedWindowComposer()
{
	Clear();
}

void SegmentBasedWindowComposer::Clear()
{
	sp_comp_features.clear();
	innerSegs.clear();	// clear pointers
}

bool SegmentBasedWindowComposer::Init(const Mat& seg_map, const vector<SegSuperPixelFeature>& sp_features, const Mat& bg_weight_map)
{
	//Clear();	no need to clear again: call once in detector::init
	
	// init data
	m_nImgWidth = seg_map.cols;
	m_nImgHeight = seg_map.rows;
	seg_index_map = seg_map;
	compose_cost_map.Create(m_nImgWidth, m_nImgHeight);
	m_fMaxCost = 0.7f;

#ifdef RECORD_AUXILIARY_INFO
	compose_cost_map.Create(m_nImgWidth, m_nImgHeight);
#endif


	//////////////////////////////////////////////////////////////////////////
	// compute composition features for each superpixel
	//////////////////////////////////////////////////////////////////////////
	sp_comp_features.resize(sp_features.size());

	for (size_t i=0; i<sp_comp_features.size(); i++)
	{
		// set common data
		sp_comp_features[i].box = sp_features[i].box;
		sp_comp_features[i].centroid = sp_features[i].centroid;

		// boundary area extension
		sp_comp_features[i].extendedArea = sp_features[i].area;
		if(sp_features[i].bnd_pixels > 0)
			sp_comp_features[i].extendedArea *= (1 + 2*(float)sp_features[i].bnd_pixels / sp_features[i].perimeter);
		
		// create mask integral
		sp_comp_features[i].CreateAreaIntegral(seg_map, i);
	}

	// compute background weight for each superpixel
	if( !bg_weight_map.empty() )
	{
		for(size_t i=0; i<sp_features.size(); i++)
		{
			Rect box = sp_features[i].box;
			sp_comp_features[i].importWeight = 0;
			for(int y=box.y; y<box.br().y; y++)
			{
				for(int x=box.x; x<box.br().x; x++)
				{
					if(seg_map.at<int>(x, y) == i)
					{
						sp_comp_features[i].importWeight += bg_weight_map.at<float>(x,y);
					}
				}
			}
			sp_comp_features[i].importWeight /= sp_features[i].area;
			
			// if the cost is above some threshold, set the pixel weight to 1 to avoid over-focusing locally
			if(sp_comp_features[i].importWeight > 0.3f)	
				sp_comp_features[i].importWeight = 1.0f;
		}
	}

	// compute pair-wise composition cost
	float maxAppdist = 0, maxSpadist = 0;
	for (int curIdx = 0; curIdx < sp_features.size(); curIdx++)
	{	
		// allocate space
		sp_comp_features[curIdx].pairs.reserve(sp_comp_features.size());

		// temp object
		TPair pair;
		for (int nextIdx = 0; nextIdx < sp_features.size(); nextIdx++)
		{			
			pair.id = nextIdx;

			float colordist = 0, depthdist = 0;
			// appearance distance
			if( (use_feat & SAL_COLOR) != 0 )
				colordist = SegSuperPixelFeature::FeatureIntersectionDistance(sp_features[curIdx], sp_features[nextIdx], SAL_COLOR);
			if( (use_feat & SAL_DEPTH) != 0 )
				depthdist = SegSuperPixelFeature::FeatureIntersectionDistance(sp_features[curIdx], sp_features[nextIdx], SAL_DEPTH);
			
			pair.appdist = (colordist + depthdist) / 2;

			if (pair.appdist > maxAppdist)
				maxAppdist = pair.appdist;

			// spatial distance
			pair.spadist = HausdorffDist(sp_features[curIdx].box, sp_features[nextIdx].box);

			if (pair.spadist > maxSpadist)
				maxSpadist = pair.spadist;

			// add to list
			sp_comp_features[curIdx].pairs.push_back(pair);
		}
	}
	// normalize
	for (int n = 0; n < sp_comp_features.size(); n++)
	{
		for(size_t i = 0; i < sp_comp_features[n].pairs.size(); i++)
		{			
			sp_comp_features[n].pairs[i].appdist /= maxAppdist;
			sp_comp_features[n].pairs[i].spadist /= maxSpadist;
		}
	}
	// sort all lists
	for (int n = 0; n < sp_comp_features.size(); n++)
	{
		SegSuperPixelComposeFeature& curfeat = sp_comp_features[n];
		// compute distance(similarity) with other segments
		for(size_t pi = 0; pi < curfeat.pairs.size(); pi++)
		{			
			float weight = curfeat.pairs[pi].spadist;

			// set saliency value for each superpixel
			curfeat.pairs[pi].saliency = (1-weight)*curfeat.pairs[pi].appdist + weight*1;
		}

		// sort by saliency
		sort(curfeat.pairs.begin(), curfeat.pairs.end(), TPair::comp_by_saliency);
	}

	return true;
}

inline float SegmentBasedWindowComposer::compose_greedy(const Rect& win)
{
	// compose order
	sort(innerSegs.begin(), innerSegs.end(), SegSuperPixelComposeFeature::comp_by_dist);
	
	float rate = 0;	// compute weight ratio to penalize background window
	float winscore = 0;

#ifdef RECORD_AUXILIARY_INFO
	compose_cost_map.FillPixels(0);
#endif

	for (size_t i = 0; i < innerSegs.size(); i++)
	{		
		SegSuperPixelComposeFeature& curfeat = *innerSegs[i];
		rate += curfeat.leftInnerArea * curfeat.importWeight;

#ifdef RECORD_AUXILIARY_INFO
		curfeat.composers.clear();
#endif

		// use all other segments to fill this one in order of their similarity to this one
		curfeat.composition_cost = 0;
		for (size_t pi = 0; pi < curfeat.pairs.size(); pi++)
		{
			SegSuperPixelComposeFeature& feat = sp_comp_features[curfeat.pairs[pi].id];
			// don't use itself, this condition is redundant and implied in the one below
			//if (curfeat.pairs[pi].id == innerSegs[i].id) continue;
			if (feat.leftOuterArea <= 0) continue;

			float fillarea = 0;

			if (curfeat.leftInnerArea <= feat.leftOuterArea)	//enough
			{
				fillarea = curfeat.leftInnerArea;
				curfeat.leftInnerArea = 0;
				feat.leftOuterArea -= curfeat.leftInnerArea;
			}
			else
			{
				fillarea = feat.leftOuterArea;
				curfeat.leftInnerArea -= feat.leftOuterArea;
				feat.leftOuterArea = 0;
			}

			curfeat.composition_cost += curfeat.pairs[pi].saliency * fillarea * curfeat.importWeight;

#ifdef RECORD_AUXILIARY_INFO
			// add composer
			curfeat.composers.push_back(curfeat.pairs[pi]);
#endif

			if (curfeat.leftInnerArea <= 0)	//finish
				break;
		}

		if (curfeat.leftInnerArea > 0)	// fill with maximum distance 1
		{				
			curfeat.composition_cost += m_fMaxCost * curfeat.leftInnerArea;
			curfeat.leftInnerArea = 0;
		}
		
		winscore += curfeat.composition_cost;	// MODIFIED

#ifdef RECORD_AUXILIARY_INFO
		// set composition cost map
		for(size_t i=0; i<curfeat.mask.size(); i++)
		{
			Point loc(i%curfeat.box.width, i/curfeat.box.width);
			if(curfeat.mask[i] > 0)
				compose_cost_map.Pixel(loc.X, loc.Y) = curfeat.composition_cost;
		}
#endif
	}

	const float winarea_inv = 1.0f / (win.width*win.height);
	rate *= winarea_inv;
	return winscore * winarea_inv * rate;

}

float SegmentBasedWindowComposer::Compose(const Rect& win)
{	
	innerSegs.clear();
	innerSegs.reserve( sp_comp_features.size() );

	// init all segments' inside and outside area with respect to the window
	// and init prepare all inside segments
	Point2f winCenter(win.x+win.width/2, win.y+win.height/2);

	for(size_t i=0; i<sp_comp_features.size(); i++)
	{
		// initialize area
		SegSuperPixelComposeFeature& curfeat = sp_comp_features[i];
		float innerArea = curfeat.AreaIn(win);

		// reset composition cost
		curfeat.composition_cost = 0;

		if (innerArea > 0)
		{
			curfeat.InitFillArea(innerArea);
			// prepare inside segments
			if (curfeat.leftInnerArea > 0)
			{
				curfeat.dist_to_win = l2_dist(sp_comp_features[i].centroid, winCenter);
				innerSegs.push_back(&curfeat);
			}
		}
		else
			curfeat.InitFillArea();
	}

	return compose_greedy(win);

}

void SegmentBasedWindowComposer::ComposeAll(const int win_width, const int win_height, const bool use_ehsw)
{		
	compose_cost_map.FillPixels(0);

	ScoredRect win(Rect(0, 0, win_width, win_height));
	for(win.y = 0; win.y < m_nImgHeight - win.height+1; win.y++)
	{
		for(win.x = 0; win.x < m_nImgWidth - win.width+1; win.x++)
		{				
			win.score = Compose(win);
			compose_cost_map.Pixel(win.x+win.width/2, win.y+win.height/2) = win.score;
		}
	}
}