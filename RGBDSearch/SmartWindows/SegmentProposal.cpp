#include "SegmentProposal.h"


SegmentProposal::SegmentProposal(void)
{
}

//////////////////////////////////////////////////////////////////////////

bool SegmentProposal::SegmentDepth(const Mat& dmap)
{
	// normalize dmap
	Mat dmap_norm;
	normalize(dmap, dmap_norm, 1, 0, CV_MINMAX);



	return true;
}
