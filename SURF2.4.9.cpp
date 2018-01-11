#include "SURF2.4.9.h"

void mySURF(cv::Mat &Img1, cv::Mat &Img2, std::vector<cv::Point2f> &features1, std::vector<cv::Point2f> &features2){

	int minHessian = 400;
	cv::SurfFeatureDetector detector(minHessian);
	std::vector<cv::KeyPoint> keys1, keys2;
	detector.detect(Img1, keys1);
	detector.detect(Img2, keys2);
	cv::SurfDescriptorExtractor extractor;
	cv::Mat surfdescriptors1, surfdescriptors2;
	extractor.compute(Img1, keys1, surfdescriptors1);
	extractor.compute(Img2, keys2, surfdescriptors2);
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > surfmatches;
	matcher.match(surfdescriptors1, surfdescriptors2, surfmatches);
	double max = 0; double min = 100;
	for (int i = 0; i < surfdescriptors1.rows; i++)
	{
		double dist = surfmatches[i].distance;
		if (dist < min) min = dist;
		if (dist > max) max = dist;
	}

	for (int i = 0; i < surfdescriptors1.rows; i++)
	{
		if (surfmatches[i].distance < 0.1*max)
		{
			features1.push_back(keys1[surfmatches[i].queryIdx].pt);
			features2.push_back(keys2[surfmatches[i].trainIdx].pt);
		}
	}
}

void GlobalOutLinerRejector(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2){

	vector<cv::Point2f> temp1, temp2;
	temp1.resize(features_img1.size());
	temp2.resize(features_img2.size());
	for (int i = 0; i<features_img1.size(); i++){
		temp1[i] = features_img1[i];
		temp2[i] = features_img2[i];
	}
	features_img1.clear();
	features_img2.clear();

	bool flag = false;
	for (double shredhold = 2.0; shredhold <= 10.0; shredhold += 1.0){
		vector<uchar> mask;
		cv::findHomography(cv::Mat(temp1), cv::Mat(temp2), mask, CV_RANSAC, shredhold);
		int cc = 0;
		for (int k = 0; k<mask.size(); k++)if (mask[k] == 1)cc++;
		if (cc>0.8*mask.size()){
			for (int k = 0; k<mask.size(); k++){
				if (mask[k] == 1){
					features_img1.push_back(temp1[k]);
					features_img2.push_back(temp2[k]);
				}
			}
			flag = true;
			break;
		}
	}

	if (!flag){
		features_img1.resize(temp1.size());
		features_img2.resize(temp1.size());
		for (int k = 0; k<temp1.size(); k++){
			features_img1[k] = temp1[k];
			features_img2[k] = temp2[k];
		}
	}
}

void GlobalOutLinerRejectorOneIteration(vector<cv::Point2f> &features_img1, vector<cv::Point2f> &features_img2){

	if (features_img1.size()>10){
		vector<cv::Point2f> temp1, temp2;
		temp1.resize(features_img1.size());
		temp2.resize(features_img2.size());
		for (int i = 0; i<features_img1.size(); i++){
			temp1[i] = features_img1[i];
			temp2[i] = features_img2[i];
		}
		features_img1.clear();
		features_img2.clear();

		vector<uchar> mask;
		cv::findHomography(cv::Mat(temp1), cv::Mat(temp2), mask, CV_RANSAC);

		for (int k = 0; k<mask.size(); k++){
			if (mask[k] == 1){
				features_img1.push_back(temp1[k]);
				features_img2.push_back(temp2[k]);
			}
		}
	}
	else{
		return;
	}
}