// video_grabcut_with_sampling.cpp : Defines the entry point for the console application.
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

#define RATIO 2 // frame_resolution / sample_resolution

int main() {
    cv::VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(0))
        return 0;
    cv::Mat frame, sample;
    cap >> frame; // For determining the dimension of the captured video.
    cv::Size sampleSize(frame.cols / RATIO, frame.rows / RATIO);

    // define bounding rectangle 
    cv::Rect rectangle(50, 10, 220, 230);

    cv::Mat mask(sampleSize.height, sampleSize.width, CV_8U); // segmentation mask (4 possible values)
    mask.setTo(cv::GC_PR_BGD);

    float fg_rect1_ratio_h = 0.8;
    float fg_rect1_ratio_w = 0.1;
    float fg_rect2_ratio_h = 0.2;
    float fg_rect2_ratio_w = 0.5;
    cv::Rect fg_rect1(mask.cols / 2 - mask.cols * fg_rect1_ratio_w / 2,
        mask.rows * (1 - fg_rect1_ratio_h),
        mask.cols * fg_rect1_ratio_w,
        mask.rows * fg_rect1_ratio_h);
    cv::Rect fg_rect2(mask.cols / 2 - mask.cols * fg_rect2_ratio_w / 2,
        mask.rows * (1 - fg_rect2_ratio_h),
        mask.cols * fg_rect2_ratio_w,
        mask.rows * fg_rect2_ratio_h);
    cv::Mat fg_area1 = mask(fg_rect1); // NOTE this is reference to subarray
    cv::Mat fg_area2 = mask(fg_rect2);
    // mark as foreground
    fg_area1.setTo(cv::GC_FGD);
    fg_area2.setTo(cv::GC_FGD);

    float bg_rect1_ratio_h = 0.5; // top-left bg
    float bg_rect1_ratio_w = 0.3;
    float bg_rect2_ratio_h = 0.5; // top-right bg
    float bg_rect2_ratio_w = 0.3;
    float bg_rect3_ratio_h = 0.5; // bottom-left bg
    float bg_rect3_ratio_w = 0.05;
    float bg_rect4_ratio_h = 0.5; // bottom-right bg
    float bg_rect4_ratio_w = 0.05;
    cv::Rect bg_rect1(0,
        0,
        mask.cols * bg_rect1_ratio_w,
        mask.rows * bg_rect1_ratio_h);
    cv::Rect bg_rect2(mask.cols - mask.cols * bg_rect2_ratio_w,
        0,
        mask.cols * bg_rect2_ratio_w,
        mask.rows * bg_rect2_ratio_h);
    cv::Rect bg_rect3(0,
        mask.rows - mask.rows * bg_rect3_ratio_h,
        mask.cols * bg_rect3_ratio_w,
        mask.rows * bg_rect3_ratio_h);
    cv::Rect bg_rect4(mask.cols - mask.cols * bg_rect4_ratio_w,
        mask.rows - mask.rows * bg_rect4_ratio_h,
        mask.cols * bg_rect4_ratio_w,
        mask.rows * bg_rect4_ratio_h);
    cv::Mat bg_area1 = mask(bg_rect1);
    cv::Mat bg_area2 = mask(bg_rect2);
    cv::Mat bg_area3 = mask(bg_rect3);
    cv::Mat bg_area4 = mask(bg_rect4);
    // mark as background
    bg_area1.setTo(cv::GC_BGD);
    bg_area2.setTo(cv::GC_BGD);
    bg_area3.setTo(cv::GC_BGD);
    bg_area4.setTo(cv::GC_BGD);
    // Stretched rectangles for drawing on result
    cv::Rect fg_rect1_stretched;
    cv::Rect fg_rect2_stretched;
    cv::Rect bg_rect1_stretched;
    cv::Rect bg_rect2_stretched;
    cv::Rect bg_rect3_stretched;
    cv::Rect bg_rect4_stretched;
    fg_rect1_stretched.x = fg_rect1.x  * RATIO;
    fg_rect1_stretched.y = fg_rect1.y  * RATIO;
    fg_rect1_stretched.height = fg_rect1.height  * RATIO;
    fg_rect1_stretched.width = fg_rect1.width  * RATIO;
    fg_rect2_stretched.x = fg_rect2.x  * RATIO;
    fg_rect2_stretched.y = fg_rect2.y  * RATIO;
    fg_rect2_stretched.height = fg_rect2.height  * RATIO;
    fg_rect2_stretched.width = fg_rect2.width  * RATIO;
    bg_rect1_stretched.x = bg_rect1.x  * RATIO;
    bg_rect1_stretched.y = bg_rect1.y  * RATIO;
    bg_rect1_stretched.height = bg_rect1.height  * RATIO;
    bg_rect1_stretched.width = bg_rect1.width  * RATIO;
    bg_rect2_stretched.x = bg_rect2.x  * RATIO;
    bg_rect2_stretched.y = bg_rect2.y  * RATIO;
    bg_rect2_stretched.height = bg_rect2.height  * RATIO;
    bg_rect2_stretched.width = bg_rect2.width  * RATIO;
    bg_rect3_stretched.x = bg_rect3.x  * RATIO;
    bg_rect3_stretched.y = bg_rect3.y  * RATIO;
    bg_rect3_stretched.height = bg_rect3.height  * RATIO;
    bg_rect3_stretched.width = bg_rect3.width  * RATIO;
    bg_rect4_stretched.x = bg_rect4.x  * RATIO;
    bg_rect4_stretched.y = bg_rect4.y  * RATIO;
    bg_rect4_stretched.height = bg_rect4.height  * RATIO;
    bg_rect4_stretched.width = bg_rect4.width  * RATIO;
    cv::Mat bgModel, fgModel; // the models (internally used by algorithm)

    while (true) {
        cap >> frame;
        if (frame.empty()) break; // end of video stream
        cv::resize(frame, sample, sampleSize);
        /*cv::putText(frame,
        text.c_str(),
        cv::Point(pos_x, pos_y), // Coordinates
        cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
        2.0, // Scale 2x bigger
        cv::Scalar(255, 255, 255), // Color
        1, // Thickness
        CV_AA); // Anti-alias
        */
        cv::grabCut(sample,    // input sample of the original frame
            mask,   // mask and segmentation result
            rectangle,// rectangle containing foreground 
            bgModel, fgModel, // models
            1,        // number of iterations
            cv::GC_INIT_WITH_MASK); // use mask
                                    // Get all foreground and possible foreground pixels
        cv::Mat result = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);

        // For comparison Display
        cv::Mat foreground_sample(sample.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        sample.copyTo(foreground_sample, result); // bg pixels not copied

        // Stretch the resulting mask to original resolution
        cv::resize(result, result, frame.size());
        // Generate output image
        cv::Mat foreground(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        frame.copyTo(foreground, result); // bg pixels not copied
        

        // draw rectangles
        cv::rectangle(foreground_sample, fg_rect1, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground_sample, fg_rect2, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground_sample, bg_rect1, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground_sample, bg_rect2, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground_sample, bg_rect3, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground_sample, bg_rect4, cv::Scalar(0, 0, 255), 1);
        //cv::rectangle(foreground, rectangle_stretched, cv::Scalar(255, 255, 0), 1); // Outer rectangle
        cv::rectangle(foreground, fg_rect1_stretched, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground, fg_rect2_stretched, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground, bg_rect1_stretched, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground, bg_rect2_stretched, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground, bg_rect3_stretched, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground, bg_rect4_stretched, cv::Scalar(0, 0, 255), 1);

        // display result
        cv::imshow("video cut sample", foreground_sample);
        cv::imshow("video cut", foreground);
        if (cv::waitKey(1) == 27) break; // press ESC to exit
    }
    return 0;
}

