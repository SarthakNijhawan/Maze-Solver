/*
 * Processor.h
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <math.h>
#include <vector>

#include "image_pool.h"

#define DO_BIN 0
#define DO_THIN 1
#define DO_SOLVE 2
#define DETECT_MSER 3
using namespace cv;

class Processor
{
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
  Mat exploreEnds(Mat im, int xprev, int yprev, int x, int y);
  Mat prune(Mat im, int sy, int sx, int ey, int ex);
  Mat extractWeirdL(Mat im,int i,int j);
  double euclidDistance(int y1, int x1, int y2, int x2);
  std::vector<cv::Point> pruneStartEnd(std::vector<cv::Point> deadCopy);
  std::vector<cv::Point> rankedExtraction(std::vector<cv::Point> deadCopy, Mat mask);
  bool checkTemplate(cv::Mat im, int i, int j, int t);
  bool edgeMatch(cv::Mat im, int i, int j, int t);
  bool cornerMatch(cv::Mat im, int i, int j, int t);
  bool isT(cv::Mat im, int i, int j);
  bool isL(cv::Mat im, int i, int j);
  bool isTetris(cv::Mat im, int i, int j);
  int connectivityNumber(cv::Mat im, int i, int j);
  int neighbors(cv::Mat result, int curX, int curY);
  int svalue(cv::Mat im, int i, int j);
  int neighbors4(cv::Mat im, int i, int j);
  Mat detectStartEndRegions(cv::Mat im);
private:
  
  bool frameAlreadyProcessed; //flag for reading in first image only
    cv::Mat curDisplay; //display stored when frameAlreadyProcessed is true
    cv::Mat curDisplayGrey;
    cv::Mat curDisplayColor;
    cv::Mat capturedMaze;
    cv::Mat intermediate1;
    cv::Mat intermediate2;
    cv::Mat result;
    cv::Mat Binimage;
    cv::MserFeatureDetector mserd;
	std::vector<cv::Point> whitepixels;
	std::vector<cv::Point> glocombcont;
    std::vector<cv::Point> deadends;
    std::vector<cv::Point> junctions;
    std::vector<int> jneighbors;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<std::vector<cv::Point2f> > imagepoints;
};

#endif /* PROCESSOR_H_ */
