// video_capture.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

//#define DEBUG  // Comment this line if building release version.

#ifdef DEBUG
//debug:
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
#endif
#ifndef DEBUG
//release:
#pragma comment(lib, "opencv_calib3d2413.lib")
#pragma comment(lib, "opencv_contrib2413.lib")
#pragma comment(lib, "opencv_core2413.lib")
#pragma comment(lib, "opencv_features2d2413.lib")
#pragma comment(lib, "opencv_flann2413.lib")
#pragma comment(lib, "opencv_gpu2413.lib")
#pragma comment(lib, "opencv_highgui2413.lib")
#pragma comment(lib, "opencv_imgproc2413.lib")
#pragma comment(lib, "opencv_legacy2413.lib")
#pragma comment(lib, "opencv_ml2413.lib")
#pragma comment(lib, "opencv_nonfree2413.lib")
#pragma comment(lib, "opencv_objdetect2413.lib")
#pragma comment(lib, "opencv_ocl2413.lib")
#pragma comment(lib, "opencv_photo2413.lib")
#pragma comment(lib, "opencv_stitching2413.lib")
#pragma comment(lib, "opencv_superres2413.lib")
#pragma comment(lib, "opencv_ts2413.lib")
#pragma comment(lib, "opencv_video2413.lib")
#pragma comment(lib, "opencv_videostab2413.lib")
#endif

#include <opencv2/opencv.hpp>

int main() {
	cv::VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
	if (!cap.open(0))
		return 0;

	cv::Mat frame, sample;
	cap >> frame; // For determining the dimension of the captured video.
	// define bounding rectangle 
	cv::Rect rectangle(50, 10, 220, 230);

	cv::Mat mask; // segmentation mask (4 possible values)
	cv::Mat bgModel, fgModel; // the models (internally used by algorithm)
	cv::Size sampleSize(320, 240);

	while (true) {
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		/*cv::putText(frame,
			text.c_str(),
			cv::Point(pos_x, pos_y), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			2.0, // Scale 2x bigger
			cv::Scalar(255, 255, 255), // Color
			1, // Thickness
			CV_AA); // Anti-alias
		*/
		cv::resize(frame, sample, sampleSize);
								  // GrabCut segmentation
		cv::grabCut(sample,    // input image
			mask,   // mask and segmentation result
			rectangle,// rectangle containing foreground 
			bgModel, fgModel, // models
			1,        // number of iterations
			cv::GC_INIT_WITH_RECT); // use rectangle
									// Get the pixels marked as likely foreground
		cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
		// Generate output image
		cv::Mat foreground(sample.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		sample.copyTo(foreground, mask); // bg pixels not copied

		//cv::rectangle(foreground, rectangle, cv::Scalar(255, 255, 0), 1);

		// display result
		cv::imshow("video cut", foreground);
		if (cv::waitKey(1) == 27) break; // press ESC to exit
	}
	return 0;
}

