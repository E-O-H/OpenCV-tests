// final.cpp : Defines the entry point for the console application.
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
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#define RATIO 2 // frame_resolution / sample_resolution
#define FACE_RATIO_X 0.55 // foreground rectangle / facial recognition output rectangle
#define FACE_RATIO_Y 1.3
#define FACE_MOVE_THRESH 32
#define FACE_BG_DIST 32
#define HAAR_CASCADE_PATH ".\\haarcascade_frontalface_alt.xml" // path of the pre-trained haarcascade.
#define BACKGROUND_PATH ".\\background.png" // path of the pre-trained haarcascade.

int main() {
    cv::VideoCapture cap;
    int deviceId = 0;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(deviceId)) {
        std::cerr << "Capture Device ID " << deviceId << "cannot be opened." << std::endl;
        return -1;
    }
    cv::Mat frame, sample;
    cap >> frame; // For determining the dimension of the captured video.
    cv::Size sampleSize(frame.cols / RATIO, frame.rows / RATIO);

    // define bounding rectangle 
    cv::Rect rectangle(50, 10, 220, 230);

    cv::Mat mask(sampleSize.height, sampleSize.width, CV_8U); // segmentation mask (4 possible values)
    mask.setTo(cv::GC_PR_BGD);

    float fg_rect1_ratio_h = 0.8;
    float fg_rect1_ratio_w = 0.1;
    float fg_rect2_ratio_h = 0.15;
    float fg_rect2_ratio_w = 0.4;
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
    /*fg_area1.setTo(cv::GC_FGD);*/
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
    /*bg_area1.setTo(cv::GC_BGD);
    bg_area2.setTo(cv::GC_BGD);*/
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

    // Load background image
    cv::Mat background = cv::imread(BACKGROUND_PATH);
    cv::resize(background, background, frame.size()); // resize to camera feed dimention

    cv::Mat foreground(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));

    // Initialize facial recognition
    std::string fn_haar = HAAR_CASCADE_PATH;
    cv::CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    while (true) {
        cap >> frame;
        if (frame.empty()) break; // end of video stream
        cv::resize(frame, sample, sampleSize);

        // Convert the current frame to grayscale:
        cv::Mat gray;
        cv::cvtColor(sample, gray, CV_BGR2GRAY);
        // Find the faces
        cv::vector< cv::Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);

        cv::Mat mask_mod = mask.clone();
        cv::Rect userFace;
        bool faceUpdated = false;
        if (!faces.empty()) {
            userFace = *std::max_element(faces.begin(), faces.end(),
                                        [](auto a, auto b) {return a.area() < b.area(); });
            cv::Rect fg_rect1_mod(std::max(userFace.x + 0.5 * (1 - FACE_RATIO_X) * userFace.width, 0.),
                                  std::max(userFace.y + 0.5 * (1 - FACE_RATIO_Y) * userFace.width, 0.),
                                  userFace.width * FACE_RATIO_X,
                                  mask.rows - std::max(userFace.y + 0.5 * (1 - FACE_RATIO_Y) * userFace.width, 0.));
            if (abs(fg_rect1_mod.x - fg_rect1.x) < FACE_MOVE_THRESH &&
                abs(fg_rect1_mod.y - fg_rect1.y) < FACE_MOVE_THRESH) {
                faceUpdated = true;
                fg_rect1 = fg_rect1_mod;
                
                bg_rect1.width = std::max(userFace.x - FACE_BG_DIST, 0);
                bg_rect2.width = std::max(mask.cols - userFace.x - userFace.width - FACE_BG_DIST, 0);
                bg_rect2.x = mask.cols - bg_rect2.width;
            }
        }
        fg_area1 = mask_mod(fg_rect1);
        fg_area1.setTo(cv::GC_FGD);

        bg_area1 = mask_mod(bg_rect1);
        bg_area2 = mask_mod(bg_rect2);
        bg_area1.setTo(cv::GC_BGD);
        bg_area2.setTo(cv::GC_BGD);

        cv::grabCut(sample,    // input sample of the original frame
            mask_mod,   // mask and segmentation result
            rectangle,// rectangle containing foreground 
            bgModel, fgModel, // models
            1,        // number of iterations
            cv::GC_INIT_WITH_MASK); // use mask
        // Get all foreground and possible foreground pixels
        cv::Mat result = (mask_mod == cv::GC_FGD) | (mask_mod == cv::GC_PR_FGD);

        // For comparison Display
        cv::Mat foreground_sample(sample.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        sample.copyTo(foreground_sample, result); // bg pixels not copied

        // Stretch the resulting mask to original resolution
        cv::resize(result, result, frame.size());
        // Generate output image
        background.copyTo(foreground);
        frame.copyTo(foreground, result); // bg pixels not copied

        // draw rectangles
        cv::rectangle(foreground_sample, fg_rect1, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground_sample, fg_rect2, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground_sample, bg_rect1, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground_sample, bg_rect2, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground_sample, bg_rect3, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground_sample, bg_rect4, cv::Scalar(0, 0, 255), 1);
        //cv::rectangle(foreground, rectangle_stretched, cv::Scalar(255, 255, 0), 1); // Outer rectangle
        /*cv::rectangle(foreground, fg_rect1_stretched, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground, fg_rect2_stretched, cv::Scalar(255, 255, 0), 1);
        cv::rectangle(foreground, bg_rect1_stretched, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground, bg_rect2_stretched, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground, bg_rect3_stretched, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(foreground, bg_rect4_stretched, cv::Scalar(0, 0, 255), 1);*/
        for (auto face : faces) {
            cv::rectangle(foreground_sample, face, CV_RGB(255, 0, 0), 1);
            int pos_x = std::max(face.tl().x - 10, 0);
            int pos_y = std::max(face.tl().y - 10, 0);
            cv::putText(foreground_sample, "HERE IS A FACE", cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
        }
        if (faceUpdated) {
            cv::rectangle(foreground_sample, userFace, CV_RGB(0, 0, 255), 3);
        }
        // display result
        cv::imshow("Internal processing", foreground_sample);
        cv::imshow("Output", foreground);
        cv::imshow("Original sample", sample);
        if (cv::waitKey(1) == 27) break; // press ESC to exit
    }
    return 0;
}

