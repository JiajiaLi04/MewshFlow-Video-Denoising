#pragma once
#include <opencv2/core/core.hpp>     
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <iostream>

using namespace std;

void mySURF(cv::Mat &Img1, cv::Mat &Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2);

void GlobalOutLinerRejector(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2);

void GlobalOutLinerRejectorOneIteration(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2);