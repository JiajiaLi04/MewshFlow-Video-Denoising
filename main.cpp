#include <opencv2/opencv.hpp>
#include "MeshFlow.h"
#include <time.h>

#define N 4
#define COL 1920
#define ROW 1080

vector<cv::Mat> getFrame(){
	printf("Read Video\n");
	cv::VideoCapture capture("test.avi");
	// check if video successfully opened
	if (!capture.isOpened())
	{
		cerr << "The video can not open!";
		exit(0);
	}
	/*double m_fps = capture.get(CV_CAP_PROP_FPS);
	int m_video_height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int m_video_width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);*/
	cv::Mat frame, frame_copy; // current video frame
	
	vector<cv::Mat> dst;
	printf("extract frames...");
	int frame_count = 0;
	while (capture.read(frame))
	{
		frame_copy = frame.clone();
		dst.push_back(frame_copy);
		
		//frame_count++;
	}
	//printf("%d\n", frame_count);
	capture.release();
	printf("Video Capture Done!\n");

	return dst;
}

//void Imagewriter(cv::Mat &final_dst, cv::Mat &target, cv::Mat &dst){
void Imagewriter(cv::Mat &final_dst, cv::Mat &dst){
	for (int r = 0; r < ROW; r++){
	/*	for (int j = 0; j < COL; j++){
			final_dst.at<cv::Vec3f>(r, j)[0] = target.at<cv::Vec3f>(r, j)[0];
			final_dst.at<cv::Vec3f>(r, j)[1] = target.at<cv::Vec3f>(r, j)[1];
			final_dst.at<cv::Vec3f>(r, j)[2] = target.at<cv::Vec3f>(r, j)[2];
		}*/
		/*for (int j = COL; j < 2*COL; j++){
			final_dst.at<cv::Vec3f>(r, j)[0] = dst.at<cv::Vec3f>(r, j - COL)[0];
			final_dst.at<cv::Vec3f>(r, j)[1] = dst.at<cv::Vec3f>(r, j - COL)[1];
			final_dst.at<cv::Vec3f>(r, j)[2] = dst.at<cv::Vec3f>(r, j - COL)[2];
		}*/
		for (int j = 0; j < COL; j++){
			final_dst.at<cv::Vec3f>(r, j)[0] = dst.at<cv::Vec3f>(r, j )[0];
			final_dst.at<cv::Vec3f>(r, j)[1] = dst.at<cv::Vec3f>(r, j )[1];
			final_dst.at<cv::Vec3f>(r, j)[2] = dst.at<cv::Vec3f>(r, j )[2];
		}
	}
}

void myMultiplier(cv::Mat &temp, vector<cv::Mat> &channels, cv::Mat &Counter){
	cv::split(temp, channels);
	cv::multiply(Counter, channels.at(0), channels.at(0));
	cv::multiply(Counter, channels.at(1), channels.at(1));
	cv::multiply(Counter, channels.at(2), channels.at(2));
	cv::merge(channels, temp);
}

void myDivider(cv::Mat &temp, vector<cv::Mat> &channels, cv::Mat &Counter){
	cv::split(temp, channels);
	cv::divide(channels.at(0), Counter, channels.at(0));
	cv::divide(channels.at(1), Counter, channels.at(1));
	cv::divide(channels.at(2), Counter, channels.at(2));
	cv::merge(channels, temp);
}


void main(){

	time_t timeBegin, timeEnd;

	vector<cv::Mat> Frames;
	vector<cv::Mat> channels, channels_a;
	Frames = getFrame();

	cv::VideoWriter vw;
	int fps = 25;
	//cv::Size S = cv::Size((int)2 * COL, (int)ROW);
	cv::Size S = cv::Size((int)COL, (int)ROW);
	vw.open("results\\out.avi", CV_FOURCC('x', 'v', 'i', 'd'), fps, S);

	//memory set
	cv::Mat comparator(cv::Size(COL, ROW), CV_8UC1, cv::Scalar::all(15));
	cv::Mat Counter(cv::Size(COL, ROW), CV_8UC1);
	cv::Mat Counter_adder = cv::Mat::ones(cv::Size(COL, ROW), CV_8UC1);
	cv::Mat accumulater(cv::Size(COL, ROW), CV_32FC3);
	cv::Mat target(cv::Size(COL, ROW), CV_8UC3);
	cv::Mat temp(cv::Size(COL, ROW), CV_8UC3);
	cv::Mat temp1(cv::Size(COL, ROW), CV_8U);
	cv::Mat temp2(cv::Size(COL, ROW), CV_8U);
	cv::Mat diff(cv::Size(COL, ROW), CV_8U);
	cv::Mat final_dst = cv::Mat::zeros(S, CV_32FC3);
	cv::Mat frame;
	printf("The frames.size is %d\n", Frames.size());
	for (int i = 0; i < Frames.size(); i++){
		
		timeBegin = time(NULL);

		//cv::Mat target = Frames[i];
		Frames[i].convertTo(target, CV_32FC3);
		accumulater += target;

		for (int j = i - N; j <= i + N; j++){
			if (j >= 0 && j < Frames.size() && j != i){
				//cv::Mat source = Frames[j];

				MeshFlow meshflow(Frames[j], Frames[i]);//Frames[j]为source,Frames[i]为target
				meshflow.Execute();
				//Mesh* mesh = meshflow.GetDestinMesh();

				meshflow.new_GetWarpedSource(temp);//temp为warp后的结果

				cv::cvtColor(Frames[i], temp1, cv::COLOR_BGR2GRAY);
				cv::cvtColor(temp, temp2, cv::COLOR_BGR2GRAY);

				absdiff(temp2, temp1, diff);
				cv::compare(diff, comparator, Counter, cv::CMP_LT);
				cv::divide(Counter, 255, Counter);
				//cout << Counter << endl << endl;

				myMultiplier(temp, channels, Counter);	//add the mask.

				//Accumulate
				temp.convertTo(temp, CV_32FC3);
				accumulater += temp;
				Counter_adder += Counter;
				temp.setTo(cv::Scalar::all(0));
				temp.convertTo(temp, CV_8UC3);
			}
			else continue;
		}
		//cout << Counter_adder << endl << endl;

		Counter_adder.convertTo(Counter_adder, CV_32FC1);
		myDivider(accumulater, channels_a, Counter_adder);		//calculate the average value

		timeEnd = time(NULL);
		printf("The number %d frame time %ds\n",i,timeEnd - timeBegin);

		//Imagewriter(final_dst, target, accumulater);
		Imagewriter(final_dst, accumulater);
		final_dst.convertTo(frame, CV_8UC3);		//write the video frames.
		vw << frame;

		//printf("Frame %d done\n", i);

		accumulater.setTo(cv::Scalar::all(0));
		Counter_adder.setTo(cv::Scalar::all(1));
		Counter_adder.convertTo(Counter_adder, CV_8UC1);
	}
	vw.release();
}