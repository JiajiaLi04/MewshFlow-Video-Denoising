#include "SURF2.4.9.h"
#include "Mesh.h"

#ifndef __VECPT2F__
#define VecPt2f vector<cv::Point2f>
#endif __VECPT2F__

#define RADIUS 50

#ifndef __NODE__
#define __NODE__
struct node{
	VecPt2f features;
	VecPt2f motions;
};
#endif

 
#ifndef __MESHFLOW__
#define __MESHFLOW__
class MeshFlow{

private:
	int m_height, m_width;
	int m_quadWidth, m_quadHeight;
	int m_meshheight, m_meshwidth;

	cv::Mat m_source, m_target;
	cv::Mat m_globalHomography;
	Mesh* m_mesh;
	Mesh* m_warpedmesh;
	VecPt2f m_vertexMotion;


	node n;

private:
	void MatchSURFFeatures();
	void SpatialMedianFilter();
	void DistributeMotion2MeshVertexes_MedianFilter();
	void WarpMeshbyMotion();
	cv::Point2f Trans(cv::Mat H, cv::Point2f &pt);

public:
	MeshFlow(cv::Mat &source, cv::Mat &target);
	void Execute();
	Mesh* GetDestinMesh(){ return m_warpedmesh; }
	cv::Mat GetWarpedSource();
	void new_GetWarpedSource(cv::Mat &dst);
};
#endif