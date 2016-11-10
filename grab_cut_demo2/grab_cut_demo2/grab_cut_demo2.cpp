// grab_cut_demo2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#pragma comment(lib, "opencv_calib3d2413d.lib")
#pragma comment(lib, "opencv_contrib2413d.lib")
#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_flann2413d.lib")
#pragma comment(lib, "opencv_gpu2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "opencv_legacy2413d.lib")
#pragma comment(lib, "opencv_ml2413d.lib")
#pragma comment(lib, "opencv_nonfree2413d.lib")
#pragma comment(lib, "opencv_objdetect2413d.lib")
#pragma comment(lib, "opencv_ocl2413d.lib")
#pragma comment(lib, "opencv_photo2413d.lib")
#pragma comment(lib, "opencv_stitching2413d.lib")
#pragma comment(lib, "opencv_superres2413d.lib")
#pragma comment(lib, "opencv_ts2413d.lib")
#pragma comment(lib, "opencv_video2413d.lib")
#pragma comment(lib, "opencv_videostab2413d.lib")

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(size_t argc, char ** argv)
{
	// Open another image
	Mat image;
	image = cv::imread(argv[1]);

	// define bounding rectangle 
	cv::Rect rectangle(50, 0, image.cols - 150, image.rows - 30);

	cv::Mat result; // segmentation result (4 possible values)
	cv::Mat bgModel, fgModel; // the models (internally used)

							  // GrabCut segmentation
	cv::grabCut(image,    // input image
		result,   // segmentation result
		rectangle,// rectangle containing foreground 
		bgModel, fgModel, // models
		1,        // number of iterations
		cv::GC_INIT_WITH_RECT); // use rectangle
	// Get the pixels marked as likely foreground
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	// Generate output image
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	image.copyTo(foreground, result); // bg pixels not copied

									  // draw rectangle on original image
	cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);
	cv::namedWindow("Image");
	cv::imshow("Image", image);

	//cv::namedWindow("result");
	//cv::imshow("result", result);

	// display result
	cv::namedWindow("Segmented Image");
	cv::imshow("Segmented Image", foreground);


	waitKey();
	return 0;
}