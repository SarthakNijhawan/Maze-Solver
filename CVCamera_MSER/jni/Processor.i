/*
 * include the headers required by the generated cpp code
 */
%{
#include "Processor.h"
#include "image_pool.h"
#include "math.h"
using namespace cv;
%}


/**
 * some constants, see Processor.h
 */
#define DO_BIN 0
#define DO_THIN 1
#define DO_SOLVE 2
#define DETECT_MSER 3

//import the android-cv.i file so that swig is aware of all that has been previous defined
//notice that it is not an include....
%import "android-cv.i"

//make sure to import the image_pool as it is 
//referenced by the Processor java generated
//class
%typemap(javaimports) Processor "
import com.opencv.jni.image_pool;// import the image_pool interface for playing nice with
								 // android-opencv

/** Processor - for processing images that are stored in an image pool
*/"

class Processor {
public:
	Processor();
	virtual ~Processor();
	
	void extractAndSolveMaze(int idx, image_pool* pool, int feature_type);
	void liveFeed(int idx, image_pool* pool, int feature_type);
	void cleanAndFilterImage(int input_idx, image_pool* pool);
	cv::Mat convexLeftRight(cv::Mat im);
  	cv::Mat convexUpDown(cv::Mat im);
  	cv::Mat erodeImage(cv::Mat im, int se_cols, int se_rows);
  	cv::Mat dilateImage(cv::Mat im,int se_cols, int se_rows);
	void morphThinning();
	void morphThinning2();
	void morphThinningZS();
	void morphThinningStentiford();
	cv::Mat exploreEnds(cv::Mat im, int xprev, int yprev, int x, int y);
	cv::Mat prune(cv::Mat im, int sy, int sx, int ey, int ex);
	double euclidDistance(int y1, int x1, int y2, int x2);
	std::vector<cv::Point> pruneStartEnd(std::vector<cv::Point> deadCopy);
	std::vector<cv::Point> rankedExtraction(std::vector<cv::Point> deadCopy, cv::Mat mask);
	bool edgeMatch(cv::Mat im, int i, int j, int t);
  	bool cornerMatch(cv::Mat im, int i, int j, int t);
	bool checkTemplate(cv::Mat im, int i, int j, int t);
	bool isT(cv::Mat im, int i, int j);
	bool isL(cv::Mat im, int i, int j);
	bool isTetris(cv::Mat im, int i, int j);
	cv::Mat extractWeirdL(cv::Mat im, int i,int j);
	int connectivityNumber(cv::Mat im, int i, int j);
	int neighbors(cv::Mat result, int curX, int curY);
	int svalue(cv::Mat im, int i, int j);
	int neighbors4(cv::Mat im, int i, int j);
	cv::Mat detectStartEndRegions(cv::Mat im);
	
	
};
