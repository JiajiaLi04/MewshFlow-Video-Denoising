#include "MeshFlow.h"

MeshFlow::MeshFlow(cv::Mat &source, cv::Mat &target){
	this->m_source = source;
	this->m_target = target;

	m_height = m_source.rows;
	m_width = m_target.cols;
	m_quadWidth = 1.0*m_width / pow(2.0, 4.0);
	m_quadHeight = 1.0*m_height / pow(2.0, 4.0);

	m_mesh = new Mesh(m_height, m_width, 1.0*m_quadWidth, 1.0*m_quadHeight);
	m_warpedmesh = new Mesh(m_height, m_width, 1.0*m_quadWidth, 1.0*m_quadHeight);
	m_meshheight = m_mesh->height;
	m_meshwidth = m_mesh->width;
	m_vertexMotion.resize(m_meshheight*m_meshwidth);
}

void MeshFlow::MatchSURFFeatures(){

	std::vector<cv::Point2f> s_fea, t_fea;
	mySURF(m_source, m_target, s_fea, t_fea);

	GlobalOutLinerRejectorOneIteration(s_fea, t_fea);

	m_globalHomography = cv::findHomography(cv::Mat(s_fea), cv::Mat(t_fea), CV_LMEDS);

	n.features = t_fea;
	n.motions.resize(t_fea.size());

	for (int i = 0; i<t_fea.size(); i++){
		n.motions[i] = s_fea[i] - t_fea[i];
	}
}

void MeshFlow::Execute(){

	MatchSURFFeatures();
	DistributeMotion2MeshVertexes_MedianFilter();  //the first median filter
	SpatialMedianFilter(); //the second median filter
	WarpMeshbyMotion();
}

void MeshFlow::DistributeMotion2MeshVertexes_MedianFilter(){
	for (int i = 0; i<m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){
			cv::Point2f pt = m_mesh->getVertex(i, j);
			cv::Point2f pttrans = Trans(m_globalHomography, pt);
			m_vertexMotion[i*m_meshwidth + j].x = pt.x - pttrans.x;
			m_vertexMotion[i*m_meshwidth + j].y = pt.y - pttrans.y;
		}
	}

	vector<vector<float>> motionx, motiony;
	motionx.resize(m_meshheight*m_meshwidth);
	motiony.resize(m_meshheight*m_meshwidth);

	//distribute features motion to mesh vertexes
	for (int i = 0; i<m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){
			cv::Point2f pt = m_mesh->getVertex(i, j);

			for (int k = 0; k<n.features.size(); k++){
				cv::Point2f pt2 = n.features[k];
				float dis = sqrt((pt.x - pt2.x)*(pt.x - pt2.x) + (pt.y - pt2.y)*(pt.y - pt2.y));

				float w = 1.0 / dis;
				if (dis<RADIUS){
					motionx[i*m_meshwidth + j].push_back(n.motions[k].x);
					motiony[i*m_meshwidth + j].push_back(n.motions[k].y);
				}
			}
		}
	}

	for (int i = 0; i<m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){
			if (motionx[i*m_meshwidth + j].size()>1){
				myQuickSort(motionx[i*m_meshwidth + j], 0, motionx[i*m_meshwidth + j].size() - 1);
				myQuickSort(motiony[i*m_meshwidth + j], 0, motiony[i*m_meshwidth + j].size() - 1);
				m_vertexMotion[i*m_meshwidth + j].x = motionx[i*m_meshwidth + j][motionx[i*m_meshwidth + j].size() / 2];
				m_vertexMotion[i*m_meshwidth + j].y = motiony[i*m_meshwidth + j][motiony[i*m_meshwidth + j].size() / 2];
			}
		}
	}
}

void MeshFlow::SpatialMedianFilter(){
	VecPt2f tempVertexMotion(m_meshheight*m_meshwidth);
	for (int i = 0; i<m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){
			tempVertexMotion[i*m_meshwidth + j] = m_vertexMotion[i*m_meshwidth + j];
		}
	}

	int radius = 5;
	for (int i = 0; i<m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){

			vector<float> motionx;
			vector<float> motiony;
			for (int k = -radius; k <= radius; k++){
				for (int l = -radius; l <= radius; l++){
					if (k >= 0 && k<m_meshheight && l >= 0 && l<m_meshwidth){
						motionx.push_back(tempVertexMotion[i*m_meshwidth + j].x);
						motiony.push_back(tempVertexMotion[i*m_meshwidth + j].y);
					}
				}
			}
			myQuickSort(motionx, 0, motionx.size() - 1);
			myQuickSort(motiony, 0, motiony.size() - 1);
			m_vertexMotion[i*m_meshwidth + j].x = motionx[motionx.size() / 2];
			m_vertexMotion[i*m_meshwidth + j].y = motiony[motiony.size() / 2];
		}
	}
}

void MeshFlow::WarpMeshbyMotion(){

	for (int i = 0; i<m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){
			cv::Point2f s = m_mesh->getVertex(i, j);
			s += m_vertexMotion[i*m_meshwidth + j];
			m_warpedmesh->setVertex(i, j, s);
		}
	}
}

cv::Mat MeshFlow::GetWarpedSource(){

	cv::Mat dst = cv::Mat::zeros(m_source.size(), CV_8UC3);
	meshWarpRemap(m_source, dst, *m_mesh, *m_warpedmesh);
	return dst;
}

void MeshFlow::new_GetWarpedSource(cv::Mat &dst){
	meshWarpRemap(m_source, dst, *m_mesh, *m_warpedmesh);
}

cv::Point2f MeshFlow::Trans(cv::Mat H, cv::Point2f &pt){
	cv::Point2f result;

	double a = H.at<double>(0, 0) * pt.x + H.at<double>(0, 1) * pt.y + H.at<double>(0, 2);
	double b = H.at<double>(1, 0) * pt.x + H.at<double>(1, 1) * pt.y + H.at<double>(1, 2);
	double c = H.at<double>(2, 0) * pt.x + H.at<double>(2, 1) * pt.y + H.at<double>(2, 2);

	result.x = a / c;
	result.y = b / c;

	return result;
}