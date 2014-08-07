
#pragma once

#include "a9wins/SmartWindowing.h"
#include "common.h"

class A9Window
{
private:

public:
	A9Window(void);

	void GenerateBlocks(const cv::Mat& frame, std::vector<ImgWin>& wins);
};

