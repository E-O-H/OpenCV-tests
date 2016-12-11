// video_capture.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

//#define DEBUG  // Comment this line if building release version.
#define DELAY_FRAMES 100

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
using namespace cv;


int main(int argc, char** argv) {
	VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
	if (!cap.open(0))
		return 0;
	/*
	// text to display
	string text;
	int pos_x, pos_y;
	std::cin >> text >> pos_x >> pos_y;*/

	Mat frame, frameGrey;
	cap >> frame; // For determining the dimension of the captured video.
	// define bounding rectangle 
	cv::Rect rectangle(50, 0, frame.cols - 150, frame.rows - 30);

	cv::Mat result; // segmentation result (4 possible values)
	cv::Mat bgModel, fgModel; // the models (internally used)

	cv::Mat background(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat backgroundGrey;
	cv::Mat mask;
	bool delayCapture = false;
	int delayFrames = DELAY_FRAMES;
	char textBuffer[5];
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

		/*						  // GrabCut segmentation
		cv::grabCut(frame,    // input image
			result,   // segmentation result
			rectangle,// rectangle containing foreground 
			bgModel, fgModel, // models
			1,        // number of iterations
			cv::GC_INIT_WITH_RECT); // use rectangle
									// Get the pixels marked as likely foreground
		cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
		// Generate output image
		cv::Mat foreground(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		frame.copyTo(foreground, result); // bg pixels not copied

		cv::rectangle(foreground, rectangle, cv::Scalar(255, 255, 255), 1);
		*/

		int key = waitKey(1);
		if (key == 99) { // press "c" to initiate copying this frame to background
			delayCapture = true;
		}
		else if (key == 27) { // press "esc" to exit
			break;
		}
		if (delayCapture) {
			if (delayFrames-- > 0) {
				_itoa_s(delayFrames, textBuffer,5, 10);
				cv::putText(frame,
					textBuffer,
					cv::Point(250, 250), // Coordinates
					cv::FONT_HERSHEY_PLAIN, // Font
					10.0, // Scale 2x bigger
					cv::Scalar(255, 0, 0), // Color
					10, // Thickness
					CV_AA); // Anti-alias
			}
			else {
				frame.copyTo(background);
				cv::cvtColor(background, backgroundGrey, CV_BGR2GRAY);
				delayCapture = false;
				delayFrames = DELAY_FRAMES;
			}
		}
		// display result
		cv::cvtColor(frame, frameGrey, CV_BGR2GRAY);
		mask = abs(frameGrey - backgroundGrey); 
		    // abs is very important! Or negative values will be treated as 0 in the mask
		    // (no automatic overflow)
		cv::Mat maskBinary;
		cv::threshold(mask, maskBinary, 5, 255, cv::THRESH_BINARY);
		//cv::adaptiveThreshold(mask, maskBinary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
		cv::Mat foreground;
		frame.copyTo(foreground, maskBinary);
			// Note to self: copyTo does not overwrite the original image outside the mask. 
			// So use a new blank foreground everytime, or there will be residue from last frame.
		cv::imshow("video subtraction", foreground);
	}
	return 0;
}

