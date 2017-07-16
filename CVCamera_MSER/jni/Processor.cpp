/*
 * Processor.cpp
 * modified version 5/26/13
 */

#include "Processor.h"
#include <sys/stat.h>
#include <android/log.h>
#include <jni.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace cv;

Processor::Processor()
	// Initialize parameters for different detectors
	//stard(20/*max_size*/, 8/*response_threshold*/, 15/*line_threshold_projected*/, 8/*line_threshold_binarized*/, 5/*suppress_nonmax_size*/),
	//fastd(50/*threshold*/, true/*nonmax_suppression*/),
	//surfd(1600/*hessian_threshold*/, 3/*octaves*/, 1/*octave_layers*/),
	//mserd(10/*delta*/, 10/*min_area*/, 500/*max_area*/, 0.2/*max_variation*/, 0.7/*min_diversity*/, 2/*max_evolution*/, 1/*area_threshold*/, 5/*min_margin*/, 2/*edge_blur_size*/)
{
	frameAlreadyProcessed = false;
	curDisplay = cvCreateMat(10,10, CV_8UC1);
	curDisplayGrey = cvCreateMat(10, 10, CV_8UC1);
	curDisplayColor = cvCreateMat(10, 10, CV_8UC1);
	Binimage = cvCreateMat(10,10,CV_8UC1);
	vector<Point> whitepixels;//To avoid iterating over old
	vector<Point> glocombcont;
	vector<Point> deadends;
	vector<Point> junctions;
	vector<int> jneighbors;//Number of neighbors for a particular junction
}

Processor::~Processor()
{
  // TODO Auto-generated destructor stub
}

/*Solve maze and store intermediate images into global variables*/
void Processor::extractAndSolveMaze(int input_idx, image_pool* pool, int feature_type)
{
	// Get image from pool
	Mat greyimage = pool->getGrey(input_idx);
	Mat img = pool->getImage(input_idx);
	if (frameAlreadyProcessed == true) {
		curDisplayGrey.copyTo(greyimage);
		curDisplayColor.copyTo(img);
	} else {
		cleanAndFilterImage(input_idx, pool);
		greyimage.copyTo(result);
		greyimage.copyTo(intermediate1);
		//result=erodeImage(result,1,14);
		if(feature_type==DO_BIN){
			frameAlreadyProcessed=true;
			result.copyTo(Binimage);
			result.copyTo(curDisplayGrey);
		}
		else if(feature_type==DO_THIN){
			frameAlreadyProcessed=true;
			result.copyTo(Binimage);
			morphThinningZS();
			result.copyTo(curDisplayGrey);
		}
		else if(feature_type==DO_SOLVE){
		frameAlreadyProcessed=true;
		result.copyTo(Binimage);
		morphThinningZS();
		for(int i=1;i<(result.rows-1);i++)
						for(int j=1;j<(result.cols-1);j++)
							if((result.data[i*result.cols+j]==(uchar)255)&&(neighbors(result,i,j)==3))
								result=extractWeirdL(result,i,j);


		result.copyTo(greyimage);
		bitwise_not(greyimage,greyimage);
		result = prune(result,0,0,0,0);
		/*for(int i=0;i<whitepixels.size();i++)
					__android_log_print(ANDROID_LOG_VERBOSE,"Whitepixels" ,"%d, %d;",whitepixels[i].y,whitepixels[i].x);*/
		/*for(int i=0;i<result.rows;i++)
					for(int j=0;j<result.cols;j++)
						if(result.data[i*result.cols+j]!=(uchar)0)
							__android_log_print(ANDROID_LOG_VERBOSE,"data2","%d, %d;",i,j);*/
		/*
		/*THICKEN THE SOLUTION DON'T DELETE*/
		//greyimage.copyTo(result);
		//bitwise_not(result,result);
		//erodeImage();
		result=dilateImage(result,3,10);
		result.copyTo(curDisplayGrey);
		//bitwise_not(result,result);
		//Copy the greyimage solution to the color image (comment out when working in gray)
		int rows=img.rows;int cols=img.cols;
		for(int i=0;i<rows;i++)
			for(int j=0;j<cols;j++)
				if(result.data[i*cols+j]!=(uchar)0)
				{
					img.data[i * img.cols*3 + j * 3 + 0] = (uchar)255; //R
					img.data[i * img.cols*3 + j * 3 + 1] = (uchar)0; //G
					img.data[i * img.cols*3 + j * 3 + 2] = (uchar)0; //B
				}

		img.copyTo(curDisplayColor);
		}
	}
}
/*
 * live feed
 * ready for new maze to capture
 */
void Processor::liveFeed(int input_idx, image_pool* pool, int feature_type) {
	frameAlreadyProcessed = false;
	Mat greyimage = pool->getGrey(input_idx);
}

/*------------helper functions------------*/
//This functions prunes a binary skeleton by pixelating from the deadends to the junctions
//Does NOT count (sx,sy) and (ex,ey) as deadends!
Mat Processor::prune(Mat im, int sy, int sx, int ey, int ex){
	deadends.clear();junctions.clear();jneighbors.clear();whitepixels.clear();
	int n=0;//Neighbors
	int cols=im.cols;
	int rows=im.rows;
	int iter=0;
	bool pflag=false;
	Point p;

	//Pre-processing (Kill all Weird-L junctions, t-junctions, L-junctions, and Tetrises) (Run 10 times for convergence)


	for(int i=1;i<(rows-1);i++){
		for(int j=1;j<(cols-1);j++){
			//Check only for white pixels to reduce computation
			if(im.data[i*cols+j]==(uchar)255){
				n=neighbors(im,i,j);//<-8-Connected (Original Code)
				//n=neighbors4(im,i,j);//4-Connected
				if(isL(im,i,j)){
				im.data[i*cols+j]=(uchar)0;//Pixelate the L's
				__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","L %d",iter);
				__android_log_print(ANDROID_LOG_VERBOSE,"L","%d, %d",j,i);
				}
				//DEBUGGING
				else if(n==3&&isTetris(im,i,j)){
					im.data[i*cols+j]=(uchar)0;//Pixelate the Tetrises
					__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","Tetris %d",iter);
					__android_log_print(ANDROID_LOG_VERBOSE,"Tetris","%d, %d",j,i);
				}
				else if(isT(im,i,j)){
					im.data[i*cols+j]=(uchar)0;//Pixelate the T's
					__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","T %d",iter);
					__android_log_print(ANDROID_LOG_VERBOSE,"T","%d, %d",j,i);
				}

			}
		}
	}

	iter=0;
	int numwhite=0;
	//First Pass create the junctions and deadends lists
	for(int i=1;i<(rows-1);i++)
		for(int j=1;j<(cols-1);j++){
			//Check only for white pixels to reduce computation
			if(im.data[i*cols+j]==(uchar)255){
				numwhite++;
				//__android_log_print(ANDROID_LOG_FATAL,"Whitepixels" ,"%d, %d;",i,j);
				n=neighbors(im,i,j);//<-8-Connected (Original Code)
				//n=neighbors4(im,i,j);//4-Connected
				p=Point(j,i);
				//Only add non-start/end deadends
				//whitepixels.push_back(p);//Add to the list of white pixels
				if(n==1){
					deadends.push_back(p);
				}else if(n>2){
					junctions.push_back(p);
					jneighbors.push_back(n);
				}

			}
		}
	__android_log_print(ANDROID_LOG_VERBOSE,"Whitenum", "%d",numwhite);
	for(int i=0;i<deadends.size();i++)
		__android_log_print(ANDROID_LOG_VERBOSE,"DeadendsO:" ,"y=%d , x=%d",deadends[i].y,deadends[i].x);
	//Extract the start and end pixels using an L-Inf heuristic
	//deadends=pruneStartEnd(deadends);


	//Extract the start and end pixels using an exp distance heuristic
	deadends=rankedExtraction(deadends,detectStartEndRegions(Binimage));
		//For now, pick two random points to be the start and end
		//deadends.erase(deadends.begin()+(0));
		//deadends.erase(deadends.begin()+(deadends.size()-1));

		for(int i=0;i<deadends.size();i++)
				__android_log_print(ANDROID_LOG_VERBOSE,"DeadendsN:" ,"y=%d , x=%d",deadends[i].y,deadends[i].x);


		/*//Blue maze ONLY (with start and finish to the sides)
		sx=deadends[7].x;sy=deadends[7].y;
		ex=deadends[8].x;ey=deadends[8].y;
		deadends.erase(deadends.begin()+(7));
		deadends.erase(deadends.begin()+((8)-1));*/
		//DEBUGGING

		__android_log_print(ANDROID_LOG_VERBOSE,"I:deadends Size","%d",deadends.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"I:junctions Size","%d",junctions.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"I:jneighbors Size","%d",jneighbors.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"I:Whitepixels","%d",whitepixels.size());


	//Search by iterating through the deadends list
	int x; int y;
	while(deadends.size()>0&&iter<100){
		for(int i=0;i<deadends.size();i++){
			x=deadends[i].x;y=deadends[i].y;
			im.data[y*cols+x]=(uchar)0;//Start pixelation
			//Pixelate in the right direction for this dead end
			if(im.data[(y-1)*cols+x]==(uchar)255)
				im=exploreEnds(im,x,y,x,y-1);
			else if(im.data[(y-1)*cols+(x+1)]==(uchar)255)
				im=exploreEnds(im,x,y,x+1,y-1);
			else if(im.data[(y-0)*cols+(x+1)]==(uchar)255)
				im=exploreEnds(im,x,y,x+1,y);
			else if(im.data[(y+1)*cols+(x+1)]==(uchar)255)
				im=exploreEnds(im,x,y,x+1,y+1);
			else if(im.data[(y+1)*cols+x]==(uchar)255)
				im=exploreEnds(im,x,y,x,y+1);
			else if(im.data[(y+1)*cols+(x-1)]==(uchar)255)
				im=exploreEnds(im,x,y,x-1,y+1);
			else if(im.data[(y-0)*cols+(x-1)]==(uchar)255)
				im=exploreEnds(im,x,y,x-1,y);
			else if(im.data[(y-1)*cols+(x-1)]==(uchar)255)
				im=exploreEnds(im,x,y,x-1,y-1);
			else
				__android_log_print(ANDROID_LOG_VERBOSE,"CVCAMERA_MSER","There is an error in the deadends list! %d, %d",y,x);

		}
		//Clear the deadends (because we've pixelated them)
		deadends.clear();
		//Fix bad junctions before making the deadends list
		for(int i=0;i<junctions.size();i++){
			y=junctions[i].y;x=junctions[i].x;
			//Recalculate the number of neighbors for a particular junction
			//n=neighbors(im,y,x);
			//jneighbors[i]=n;
			//If a pixel is in an L-shape, eliminate it
			if(isL(im,y,x)){
				im.data[y*cols+x]=(uchar)0;
				__android_log_print(ANDROID_LOG_VERBOSE,"LM:Junctions","%d, %d",y,x);
				//DEBUGGING
				junctions.erase(junctions.begin()+i);
				jneighbors.erase(jneighbors.begin()+i);
				//i--;//So we don't skip over (might take this out)
				i=-1;
			}
		}

		//Check if we have any new deadends from the junctions list
		for(int i=0;i<junctions.size();i++){
			y=junctions[i].y;x=junctions[i].x;
			//Recalculate the number of neighbors for a particular junction
			n=neighbors(im,y,x);
			jneighbors[i]=n;
			__android_log_print(ANDROID_LOG_VERBOSE,"M:Junctions","%d, %d, %d",y,x,jneighbors[i]);
			if(jneighbors[i]<=2){
				if(jneighbors[i]==1)
					deadends.push_back(junctions[i]);
				junctions.erase(junctions.begin()+(i));
				jneighbors.erase(jneighbors.begin()+(i));
				//i--;//So we don't skip over (might take this out)
				i=-1;
			}
		}
		__android_log_print(ANDROID_LOG_VERBOSE,"M:Deadends Size","%d",deadends.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"M:Junctions Size","%d",junctions.size());
			iter++;
		}

		/*Iterate through the white pixels to save computational time
			for(int k=0;k<whitepixels.size();k++){
				y=whitepixels[k].y;x=whitepixels[k].x;
				n=neighbors(im,y,x);//<-8-Connected (Original Code)
				p=Point(y,x);
				if(n==1&&!(y==sy&&x==sx)&&!(y==ey&&x==ex))
					deadends.push_back(p);
				else if(n>2){
					junctions.push_back(p);
					jneighbors.push_back(n);
				}
				else
					__android_log_print(ANDROID_LOG_VERBOSE,"MX:DEBUG","%d",n);
			}

			__android_log_print(ANDROID_LOG_VERBOSE,"M:deadends Size","%d",deadends.size());
			__android_log_print(ANDROID_LOG_VERBOSE,"M:junctions Size","%d",junctions.size());
			__android_log_print(ANDROID_LOG_VERBOSE,"M:jneighbors Size","%d",jneighbors.size());
			__android_log_print(ANDROID_LOG_VERBOSE,"M:Whitepixels","%d",whitepixels.size());*/


		__android_log_print(ANDROID_LOG_VERBOSE,"F:deadends Size","%d",deadends.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"F:junctions Size","%d",junctions.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"F:jneighbors Size","%d",jneighbors.size());
		__android_log_print(ANDROID_LOG_VERBOSE,"F:Iterations","%d",iter);
		__android_log_print(ANDROID_LOG_VERBOSE,"F:Whitepixels","%d",whitepixels.size());



		return im;
}
bool Processor::isTetris(Mat im, int i, int j){
//N.B. This function only gets called if the pixel is white and has three neighbors
	bool result=false;
	int cols=im.cols;
	//Check the four rotations of Tetrises AND the mirror image flips
	//Right-handed
	if(im.data[i*cols+(j-1)]==(uchar)255&&im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[(i-1)*cols+(j+1)]==(uchar)255)
		result=true;
	else if(im.data[(i-1)*cols+(j-1)]==(uchar)255&&im.data[(i-0)*cols+(j-1)]==(uchar)255&&im.data[(i+1)*cols+(j-0)]==(uchar)255)
		result=true;
	else if(im.data[(i+1)*cols+(j-1)]==(uchar)255&&im.data[(i+1)*cols+(j-0)]==(uchar)255&&im.data[i*cols+(j+1)]==(uchar)255)
		result=true;
	else if(im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[(i-0)*cols+(j+1)]==(uchar)255&&im.data[(i+1)*cols+(j+1)]==(uchar)255)
		result=true;
	//Left-Handed
	if(im.data[i*cols+(j+1)]==(uchar)255&&im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[(i-1)*cols+(j-1)]==(uchar)255)
		result=true;
	else if(im.data[(i-1)*cols+(j+1)]==(uchar)255&&im.data[(i-0)*cols+(j+1)]==(uchar)255&&im.data[(i+1)*cols+(j-1)]==(uchar)255)
		result=true;
	else if(im.data[(i+1)*cols+(j+1)]==(uchar)255&&im.data[(i+1)*cols+(j-0)]==(uchar)255&&im.data[i*cols+(j-1)]==(uchar)255)
		result=true;
	else if(im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[(i-0)*cols+(j-1)]==(uchar)255&&im.data[(i+1)*cols+(j-1)]==(uchar)255)
		result=true;

	return result;
}

bool Processor::isT(Mat im, int i, int j){
	//N.B. This function only gets called if the pixel is white and has three neighbors
	bool result=false;
	int cols=im.cols;
	//Check the four rotations of T's
	if(im.data[i*cols+(j-1)]==(uchar)255&&im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[i*cols+(j+1)]==(uchar)255)
		result=true;
	else if(im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[(i-0)*cols+(j+1)]==(uchar)255&&im.data[(i+1)*cols+(j-0)]==(uchar)255)
		result=true;
	else if(im.data[i*cols+(j-1)]==(uchar)255&&im.data[(i+1)*cols+(j-0)]==(uchar)255&&im.data[i*cols+(j+1)]==(uchar)255)
		result=true;
	else if(im.data[(i-1)*cols+(j-0)]==(uchar)255&&im.data[(i-0)*cols+(j-1)]==(uchar)255&&im.data[(i+1)*cols+(j-0)]==(uchar)255)
		result=true;

	return result;
}
bool Processor::isL(Mat im, int i, int j){
	//N.B. This function gets called regardless
	bool result=false;
	int cols=im.cols;
	//Check the four rotations of L's
	if((im.data[(i-1)*cols+j]==(uchar)255)&&(im.data[i*cols+(j+1)]==(uchar)255)&&(im.data[(i+1)*cols+(j-1)]!=(uchar)255))
		result=true;
	else if((im.data[(i+1)*cols+j]==(uchar)255)&&(im.data[i*cols+(j+1)]==(uchar)255)&&(im.data[(i-1)*cols+(j-1)]!=(uchar)255))
		result=true;
	else if((im.data[(i+1)*cols+j]==(uchar)255)&&(im.data[i*cols+(j-1)]==(uchar)255)&&(im.data[(i-1)*cols+(j+1)]!=(uchar)255))
		result=true;
	else if((im.data[(i-1)*cols+j]==(uchar)255)&&(im.data[i*cols+(j-1)]==(uchar)255)&&(im.data[(i+1)*cols+(j+1)]!=(uchar)255))
		result=true;

	return result;
}

//This function should be run BEFORE the algorithm (pre-processing)
Mat Processor::extractWeirdL(Mat im,int i,int j){
	//Run over all non-junction, non-deadend pixels in image
	int cols=im.cols;
	__android_log_print(ANDROID_LOG_VERBOSE,"Processing a ","Weird L %d, %d", j, i);
	if((neighbors(im,i,j)==3)){
	//Match one of four templates
		if((im.data[(i-1)*cols+(j-1)]==(uchar)255)&&(im.data[(i-0)*cols+(j+1)]==(uchar)255)&&(im.data[(i+1)*cols+(j-0)]==(uchar)255)){
			im.data[i*cols+j]=(uchar)0;
			im.data[(i-1)*cols+j]=(uchar)255;
			__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","Weird L %d, %d", j, i);
		}
		else if((im.data[(i+1)*cols+(j+1)]==(uchar)255)&&(im.data[(i-0)*cols+(j-1)]==(uchar)255)&&(im.data[(i-1)*cols+(j-0)]==(uchar)255)){
			im.data[i*cols+j]=(uchar)0;
			im.data[(i-0)*cols+(j+1)]=(uchar)255;
			__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","Weird L %d, %d", j, i);

		}
		else if((im.data[(i+1)*cols+(j+1)]==(uchar)255)&&(im.data[(i-0)*cols+(j-1)]==(uchar)255)&&(im.data[(i-1)*cols+(j-0)]==(uchar)255)){
			im.data[i*cols+j]=(uchar)0;
			im.data[(i+1)*cols+j]=(uchar)255;
			__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","Weird L %d, %d", j, i);

		}
		else if((im.data[(i+1)*cols+(j-1)]==(uchar)255)&&(im.data[(i-0)*cols+(j+1)]==(uchar)255)&&(im.data[(i-1)*cols+(j-0)]==(uchar)255)){
			im.data[i*cols+j]=(uchar)0;
			im.data[(i-0)*cols+(j-1)]=(uchar)255;
			__android_log_print(ANDROID_LOG_VERBOSE,"Pruning a ","Weird L %d, %d", j, i);

		}

	}
	return im;
}
/*returns Euclidian distance between two points*/
double Processor::euclidDistance(int y1, int x1, int y2, int x2) {

double dy = y1-y2;
double dx = x1-x2;
return std::sqrt(dx*dx + dy*dy);
}

//This function extracts the most-likely start and end from the deadends list using an exp distance heuristic
vector<Point> Processor::rankedExtraction(vector<Point> deadCopy, Mat mask){
	//Create a whitepixels list from the mask
	vector<Point> white;
	vector<double> distances;
	Point p;
	int rows=mask.rows;
	int cols=mask.cols;
	for(int i=0;i<mask.rows;i++)
		for(int j=0;j<mask.cols;j++)
			if(mask.data[i*cols+j]==(uchar)255){
				p=Point(j,i);
				white.push_back(p);
			}
	//Iterate over the whitepixels and their distances to point on the deadends list
	double sum=0;
	double r0=1;//Mean length decay ~1 pixels might be a good first estimate
	for(int i=0;i<deadCopy.size();i++){
		for(int j=0;j<white.size();j++){
			sum+=std::exp(-1*(euclidDistance(deadCopy[i].y,deadCopy[i].x,white[j].y,white[j].x))/r0);
		}
		distances.push_back(sum);
		sum=0;
	}
	//DEBUGGING
	for(int i=0;i<distances.size();i++)
		__android_log_print(ANDROID_LOG_VERBOSE,"Distances","%f",distances[i]);

	//Calculate the max 2 indices and extract
	double max=-1;
	int maxi=-1;
	for(int i=0;i<distances.size();i++) {
		if(distances[i]>max){
			max=distances[i];
			maxi=i;
		}
	}
	//Extract
	deadCopy.erase(deadCopy.begin()+(maxi));
	distances.erase(distances.begin()+(maxi));
	//Repeat
	max=-1;maxi=-1;
	for(int i=0;i<distances.size();i++) {
		if(distances[i]>max){
			max=distances[i];
			maxi=i;
		}
	}
	//Extract
	deadCopy.erase(deadCopy.begin()+(maxi));
	//Return the list
	return deadCopy;
}

//EDIT: DELETE THIS FUNCTION!
//This functions extracts the most-likely start and end from the deadends list using an L-Inf distance heuristic
vector<Point> Processor::pruneStartEnd(vector<Point> deadCopy){
	vector<double> distances;//This will keep the distance measurements of each deadend
	//Calculate the mean x, and y
	double xmean=0,ymean=0;
	for(int i=0;i<deadCopy.size();i++){
		xmean+=deadCopy[i].x;
		ymean+=deadCopy[i].y;
	}
	xmean/=deadCopy.size();
	ymean/=deadCopy.size();
	//Subtract the mean and calculate L-Inf distance
	for(int i=0;i<deadCopy.size();i++){
		distances.push_back(max(abs(deadCopy[i].x-xmean),abs(deadCopy[i].y-ymean)));
	}
	//Calculate the max 2 indices and extract
	double max=-1;
			int maxi=-1;
			for(int i=0;i<distances.size();i++) {
				if(distances[i]>max){
					max=distances[i];
					maxi=i;
				}
			}
			//Extract
			deadCopy.erase(deadCopy.begin()+(maxi-1));
			distances.erase(distances.begin()+(maxi-1));
			//Repeat
			max=-1;maxi=-1;
			for(int i=0;i<distances.size();i++) {
				if(distances[i]>max){
					max=distances[i];
					maxi=i;
				}
			}
			//Extract
			deadCopy.erase(deadCopy.begin()+(maxi-1));
			//Return the list
			return deadCopy;

}
//This function will explore and prune off the pixels from a point (non-recursive)
Mat Processor::exploreEnds(Mat im, int xprev, int yprev, int x, int y){
	//Three Cases: Reach a junction, reach a dead end (pathological case), or another pixel
	bool flag=false;
	int cols=im.cols; int rows=im.rows;
	while(!flag){
	im.data[y*cols+x]=(uchar)0;
	//Junction Reached
	for(int i=0;i<junctions.size();i++)
		if(junctions[i].x==x&&junctions[i].y==y){
			im.data[y*cols+x]=(uchar)255;//Unpixelate
			//jneighbors[i]--;//Decrement number of neighbors for that junction
			/*if(jneighbors[i]<3){
				if(jneighbors[i]==1)
					deadends.push_back(junctions[i]);
				else
					flag=true;//Stop pixelating if it is not a deadend

				junctions.erase(junctions.begin()+i);
				jneighbors.erase(jneighbors.begin()+i);
				//i--;//So we don't skip over (might take this out)
				im.data[y*cols+x]=(uchar)0; //Just kidding, pixelate.

			}
			else
				flag=true;//This gets to pass*/
			//Delete from junctions list if it has less than 3 neighbors
			__android_log_print(ANDROID_LOG_VERBOSE,"Jneighbors","%d, %d, %d",y,x,jneighbors[i]);
			/*if(jneighbors[i]<3){
				junctions.erase(junctions.begin()+(i-1));
				jneighbors.erase(jneighbors.begin()+(i-1));
			}*/
			flag=true;
			continue;

		}
	//Normal Pixel (Check neighbors and move around)
	if(!flag){
		if((im.data[(y-1)*cols+x]==(uchar)255)&&!((xprev==x)&&(yprev==(y-1))))
		{
			xprev=x;yprev=y;
			x=x;y=y-1;
		}
		else if((im.data[(y-1)*cols+(x+1)]==(uchar)255)&&!((xprev==(x+1))&&(yprev==(y-1))))
		{
			xprev=x;yprev=y;
			x=x+1;y=y-1;
		}
		else if((im.data[(y-0)*cols+(x+1)]==(uchar)255)&&!((xprev==(x+1))&&(yprev==(y-0))))
		{
			xprev=x;yprev=y;
			x=x+1;y=y-0;
		}
		else if((im.data[(y+1)*cols+(x+1)]==(uchar)255)&&!((xprev==(x+1))&&(yprev==(y+1))))
		{
			xprev=x;yprev=y;
			x=x+1;y=y+1;
		}
		else if((im.data[(y+1)*cols+x]==(uchar)255)&&!((xprev==x)&&(yprev==(y+1))))
		{
			xprev=x;yprev=y;
			x=x;y=y+1;
		}
		else if((im.data[(y+1)*cols+(x-1)]==(uchar)255)&&!((xprev==(x-1))&&(yprev==(y+1))))
		{
			xprev=x;yprev=y;
			x=x-1;y=y+1;
		}
		else if((im.data[(y-0)*cols+(x-1)]==(uchar)255)&&!((xprev==(x-1))&&(yprev==(y-0))))
		{
			xprev=x;yprev=y;
			x=x-1;y=y-0;
		}
		else if((im.data[(y-1)*cols+(x-1)]==(uchar)255)&&!((xprev==(x-1))&&(yprev==(y-1))))
		{
			xprev=x;yprev=y;
			x=x-1;y=y-1;
		}
		else{//Pathological Case (This should never happen in a good skeleton)
			flag=true;
			__android_log_print(ANDROID_LOG_VERBOSE,"I hate","fucking patholgies");
		}
	}

	}
	return im;
}
void Processor::cleanAndFilterImage(int input_idx, image_pool* pool) {

		Mat greyimage = pool->getGrey(input_idx);
		//This will draw the largest contours to the image.
		//Binarize and NOT the image.
		Mat contourOutput;
		medianBlur(greyimage,greyimage,3);//Why this and why 3? Same for adaptive
		//Try increasing the contrast of the image here:
		/*double gamma=5;
		for(int i=0;i<greyimage.rows;i++)
			for(int j=0;j<greyimage.cols;j++)
				greyimage.data[i*greyimage.cols+j]=(int)(pow(((double)(greyimage.data[i*greyimage.cols+j]))/255,gamma)*255);
		*/
		//Local Adaptive Binarization
		adaptiveThreshold(greyimage, greyimage,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);
		int rows=greyimage.rows; int cols=greyimage.cols;

		//This code below is for contouring/region labeling
		greyimage.copyTo(contourOutput);
		contourOutput=255-contourOutput;//bitwise not, consider replacing
		//bitwise_not(contourOutput,contourOutput);
		vector<vector<Point> > contours;
		findContours(contourOutput,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

		//Use loop to cycle through contours and kill contours that are too small.
		//This code takes the max 2 perimeters in the contours list
		vector<double> perims(contours.size());
		for(int i=0;i<contours.size();i++) {
			perims[i]=arcLength(Mat(contours[i]),0);
		}
		double max1=-1,max2=-1;
		int max1i=-1,max2i=-1;
		for(int i=0;i<perims.size();i++) {
			if(perims[i]>max1){
				max1=perims[i];
				max1i=i;
			}
		}
		for(int i=0;i<perims.size();i++) {
			if (perims[i]>max2&&i!=max1i){
				max2=perims[i];
				max2i=i;
			}
		}
		//Creating a convex hull
		vector<Point> combcont;
		combcont.reserve(contours[max1i].size()+contours[max2i].size());
		combcont.insert( combcont.end(), contours[max1i].begin(), contours[max1i].end() );
		combcont.insert( combcont.end(), contours[max2i].begin(), contours[max2i].end() );
		glocombcont=combcont;

		//Use the convex hull idea to create a proper mask

		greyimage=255; //Make the greyimage white
		//Original Code
		/*
		vector<vector<Point> > hull(1);
		convexHull( Mat(combcont), hull[0], false );
		drawContours(greyimage,hull,-1,cv::Scalar(0),CV_FILLED);
		greyimage=255-greyimage;
		*/
		drawContours(greyimage, contours, max1i, cv::Scalar(0), CV_FILLED);
		drawContours(greyimage, contours, max2i, cv::Scalar(0), CV_FILLED);

		//These will draw the contours back onto the image and filter out via a convex combination of the contours as sets

		Mat mask;
		bitwise_or(convexLeftRight(greyimage),convexUpDown(greyimage),mask);
		bitwise_and(mask,greyimage,greyimage);


		//Extract small regions of white that aren't part of the big maze solution space
		greyimage.copyTo(contourOutput);
		findContours(contourOutput,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		vector<double> areas(contours.size());
		for(int i=0;i<contours.size();i++)
			areas[i]=contourArea(Mat(contours[i]),0);

		double max=-1;
		int maxi=-1;
		for(int i=0;i<contours.size();i++)
			if(areas[i]>max){
				maxi=i;
				max=areas[i];
			}
		greyimage.setTo(cv::Scalar(0));//Draw the contours on a black slate
		drawContours(greyimage,contours,maxi,cv::Scalar(255),CV_FILLED);

		//From combcont, create your own mask
		/*
		int x,y;
		int minw=cols+1,minh=rows+1,maxw=-1,maxh=-1;
		for(int i=0;i<glocombcont.size();i++){
			x=glocombcont[i].x;
			y=glocombcont[i].y;
			if(y<minh)
				minh=y;
			if(x<minw)
				minw=x;
			if(y>maxh)
				maxh=y;
			if(x>maxw)
				maxw=x;
		}
		//Use the parameters to create the mask form minh,minw,maxh,maxw
		int mpw=10;
		Mat mask = cvCreateMat(rows, cols, CV_8UC1);
		mask.setTo(cv::Scalar(0));
		for(int i=minh+mpw;i<maxh-mpw;i++)
			for(int j=minw+mpw;j<maxw-mpw;j++)
				mask.data[i*cols+j]=(uchar)255;
		bitwise_and(mask,greyimage,greyimage);
		__android_log_print(ANDROID_LOG_VERBOSE,"yxhwC" ,"%d, %d, %d, %d",minh,minw,maxh,maxw);
		*/
}
//This function will filter out areas in the image which are not between the maze walls (up or down)
Mat Processor::convexUpDown(Mat im){
	Mat imcopy;
	im.copyTo(imcopy);
	bool upflag=false;
	bool downflag=false;
	int rows=im.rows;
	int cols=im.cols;
	int y,x;
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++){
			//For each pixel, check whether it is in between two black areas
			y=i;x=j;upflag=false;downflag=false;
			//First check up
			while(y>=0&&!upflag){
				if(im.data[y*cols+x]==(uchar)0)
					upflag=true;
				y--;//Go up
			}
			//Then check down
			y=i;x=j;
			while(y<rows&&!downflag){
				if(im.data[y*cols+x]==(uchar)0)
					downflag=true;
				y++;//Go down
			}
			if(downflag&&upflag)//Only if pixel is in between two black regions
				imcopy.data[i*cols+j]=(uchar)255;
			else
				imcopy.data[i*cols+j]=(uchar)0;
		}

	return imcopy;
}
//This function will filter out areas in the image which are not between the maze walls (left or right)
Mat Processor::convexLeftRight(Mat im){
	Mat imcopy;
	im.copyTo(imcopy);
	bool leftflag=false;
	bool rightflag=false;
	int rows=im.rows;
	int cols=im.cols;
	int y,x;
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++){
			//For each pixel, check whether it is in between two black areas
			y=i;x=j;leftflag=false;rightflag=false;
			//First check up
			while(x>=0&&!leftflag){
				if(im.data[y*cols+x]==(uchar)0)
					leftflag=true;
				x--;//Go left
			}
			//Then check down
			y=i;x=j;
			while(x<cols&&!rightflag){
				if(im.data[y*cols+x]==(uchar)0)
					rightflag=true;
				x++;//Go right
			}
			if(leftflag&&rightflag)//Only if pixel is in between two black regions
				imcopy.data[i*cols+j]=(uchar)255;
			else
				imcopy.data[i*cols+j]=(uchar)0;
		}

	return imcopy;
}
//This operates on the result and erodes using horizontal and vertical structuring elements
Mat Processor::erodeImage(Mat im, int se_cols, int se_rows) {
	int n_iter = 1;
	int rows=im.rows;
	int cols=im.cols;
	//on p665:
	//Mat SE = getStructuringElement(MORPH_RECT, Size(1,10), Point(-1,-1));
	Mat SE_vert = cvCreateMat(se_rows, se_cols, CV_8UC1);
	Mat SE_horiz = cvCreateMat(se_cols, se_rows, CV_8UC1);
	Point pt = Point(-1,-1);
	Mat vert_only = cvCreateMat(rows,cols,CV_8UC1);
	Mat horiz_only = cvCreateMat(rows,cols,CV_8UC1);
	im.copyTo(vert_only);
	im.copyTo(horiz_only);
	//acquire maze vertical walls and maze horizontal walls
	erode(im, im, SE_vert, pt, n_iter, borderInterpolate(1, 2, 1 ), morphologyDefaultBorderValue());
	erode(im, im, SE_horiz, pt, n_iter, borderInterpolate(1, 2, 1 ), morphologyDefaultBorderValue());
	return im;
}
//This operates on the result and dilates using horizontal and vertical structuring elements
Mat Processor::dilateImage(Mat im, int se_cols, int se_rows) {
	int n_iter = 1;
	int rows=im.rows;
	int cols=im.cols;
	//on p665:
	//Mat SE = getStructuringElement(MORPH_RECT, Size(1,10), Point(-1,-1));
	Mat SE_vert = cvCreateMat(se_rows, se_cols, CV_8UC1);
	Mat SE_horiz = cvCreateMat(se_cols, se_rows, CV_8UC1);
	Point pt = Point(-1,-1);
	Mat vert_only = cvCreateMat(rows,cols,CV_8UC1);
	Mat horiz_only = cvCreateMat(rows,cols,CV_8UC1);
	im.copyTo(vert_only);
	im.copyTo(horiz_only);
	//acquire maze vertical walls and maze horizontal walls
	dilate(im, im, SE_vert, pt, n_iter, borderInterpolate(1, 2, 1 ), morphologyDefaultBorderValue());
	dilate(im, im, SE_horiz, pt, n_iter, borderInterpolate(1, 2, 1 ), morphologyDefaultBorderValue());
	return im;
}
void Processor::morphThinning() {
	Mat skeleton = cvCreateMat(result.rows, result.cols, CV_8UC1);
	Mat temp = cvCreateMat(result.rows, result.cols, CV_8UC1);
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	//TODO: do in binary not gray
	bool done;
	int iter = 0;
	double max;
	while (!done) {
		iter++;
		morphologyEx(result, temp, cv::MORPH_OPEN, element);
		bitwise_not(temp, temp);
		bitwise_and(result, temp, temp);
		bitwise_or(skeleton, temp, skeleton);
		erode(result, result, element);
		minMaxLoc(result, 0, &max);
		done = (max == 0);
		if (iter == 100) break;
	}
	skeleton.copyTo(result);
}
//This algorithm uses the Stentiford method for thinning
void Processor::morphThinningStentiford(){
	int iter=0;
	int id, jd;
	vector<Point> deleteList;
	int rows=result.rows;
	int cols=result.cols;
	int t=1;
	bool match=false;
	while(iter<20){
		for(int i=1;i<(rows-1);i++){
			for(int j=1;j<(cols-1);j++)
			{

				if(result.data[i*cols+j]==(uchar)255){
					match=checkTemplate(result,i,j,t);
				    if(match&&connectivityNumber(result,i,j)==1&&neighbors(result,i,j)!=1){
						Point pt= Point(i,j);
				    	deleteList.push_back(pt);
				    }
				}
			}
			for(int n=0;n<deleteList.size();n++){
				id=deleteList[n].y;
				jd=deleteList[n].x;
				result.data[id*cols+jd]=(uchar)0;
			}
			//Don't need to delete old values again
			deleteList.clear();
		}
		iter++;
		t++;
		if(t==5)
			t=1;
	}
}
//This algorithm uses the Zhang-Suen method for thinning on the image: result
void Processor::morphThinningZS(){
	int rows=result.rows;
    int cols=result.cols;
	//Check if binary
	bool flag=false;
	for(int i=1;i<(rows-1);i++)
				for(int j=1;j<(cols-1);j++)
					if((result.data[i*cols+j]!=(uchar)255)&&(result.data[i*cols+j]!=(uchar)0))
						flag=true;
	if(flag)
	{
		__android_log_print(ANDROID_LOG_VERBOSE,"CVCAMERA_MSER","This isn't binary to begin with...");
	}
	//
	int iter=0;
	int id, jd;
	Mat zs = cvCreateMat(result.rows, result.cols, CV_8UC1);
	zs.setTo(cv::Scalar(0));
	vector<Point> deleteList;
	int n=0;//Number of neighbors
	int s=0;//S-Value
	int n135=0;
	int n357=0;
	int n137=0;
	int n157=0;
	bool match=false;
	while(iter<100){
		for(int i=1;i<(rows-1);i++){
			for(int j=1;j<(cols-1);j++)
			{
				//Check for white pixels only
				if(result.data[i*cols+j]==(uchar)255){
					n=neighbors(result,i,j);// <-Original Code 8-Connected
					s=svalue(result,i,j);

					if((result.data[(i-1)*cols+j]==(uchar)255)&&(result.data[(i-0)*cols+(j+1)]==(uchar)255)&&(result.data[(i+1)*cols+j]==(uchar)255))
						n135=1;
					else
						n135=0;
					if((result.data[(i-0)*cols+(j+1)]==(uchar)255)&&(result.data[(i+1)*cols+(j-0)]==(uchar)255)&&(result.data[(i-0)*cols+(j-1)]==(uchar)255))
						n357=1;
					else
						n357=0;
					if((result.data[(i-1)*cols+j]==(uchar)255)&&(result.data[(i-0)*cols+(j+1)]==(uchar)255)&&(result.data[(i-0)*cols+(j-1)]==(uchar)255))
						n137=1;
					else
						n137=0;
					if((result.data[(i-1)*cols+j]==(uchar)255)&&(result.data[(i+1)*cols+(j-0)]==(uchar)255)&&(result.data[(i-0)*cols+(j-1)]==(uchar)255))
						n157=1;
					else
						n157=0;

					if(iter%2==0)
						match=((n>=2)&&(n<=6)&&(s==1)&&n135==0&&n357==0);
					else
						match=((n>=2)&&(n<=6)&&(s==1)&&n137==0&&n157==0);

				    if(match){
						//Point pt= Point(i,j);
				    	//deleteList.push_back(pt);
				    	zs.data[i*cols+j]=(uchar)255;
				    }
				}
			}
			//Delete marked pixels
			/*for(int n=0;n<deleteList.size();n++){
				id=deleteList[n].y;
				jd=deleteList[n].x;
				result.data[id*cols+jd]=(uchar)0;
			}*/

			//zs= cvCreateMat(result.rows, result.cols, CV_8UC1);
			//Don't need to delete old values again
			//deleteList.clear();
		}
		//Minkowski Set Subtraction
		bitwise_not(zs,zs);
		bitwise_and(result,zs,result);
		zs.setTo(cv::Scalar(0));
		iter++;
	}

}
//This algorithm uses a different (?) method for thinning
void Processor::morphThinning2(){
	int iter=0;
	int id, jd;
	vector<Point> deleteList;
	int rows=result.rows;
	int cols=result.cols;

	bool match=false;
	while(iter<10){
		for(int it=1;it<=8;it++){
		for(int i=1;i<(rows-1);i++){
			for(int j=1;j<(cols-1);j++)
			{

				if(result.data[i*cols+j]==(uchar)255){
					if(it<=4)
						match=edgeMatch(result,i,j,it);
					else
						match=cornerMatch(result,i,j,it%4);

				    if(match){
						Point pt= Point(i,j);
				    	deleteList.push_back(pt);
				    }
				}
			}
			for(int n=0;n<deleteList.size();n++){
				id=deleteList[n].y;
				jd=deleteList[n].x;
				result.data[id*cols+jd]=(uchar)0;
			}
			//Don't need to delete old values again
			deleteList.clear();
		}

		}
		iter++;

	}
}
//Checks if pixel matches Template (1,2,3 or 4)
bool Processor::checkTemplate(Mat im, int i, int j, int t){
	int cols=im.cols;
	if(t==1)
	{
		if((im.data[(i-1)*cols+j]==(uchar)0)&&(im.data[(i+1)*cols+j]==(uchar)255))
			return true;
		else
			return false;
	}
	else if(t==2)
	{
		if((im.data[i*cols+(j-1)]==(uchar)0)&&(im.data[i*cols+(j+1)]==(uchar)255))
			return true;
		else
			return false;
	}
	else if(t==3)
	{
		if((im.data[(i+1)*cols+j]==(uchar)0)&&(im.data[(i-1)*cols+j]==(uchar)255))
			return true;
		else
			return false;
	}
	else
	{
		if((im.data[i*cols+(j+1)]==(uchar)0)&&(im.data[i*cols+(j-1)]==(uchar)255))
			return true;
	    else
			return false;
	}
}
//Checks if the edge templates match (1,2,3 or 4)
bool Processor::edgeMatch(Mat im, int i, int j,int t){
	int cols=im.cols;
	if(t==1)//North
	{
		if((im.data[(i+1)*cols+(j-1)]==(uchar)255)&&(im.data[(i+1)*cols+(j-0)]==(uchar)255)&&(im.data[(i+1)*cols+(j+1)]==(uchar)255)&&(im.data[(i-1)*cols+(j-1)]==0)&&(im.data[(i-1)*cols+(j-0)]==0)&&(im.data[(i-1)*cols+(j+1)]==0))
			return true;
		else
			return false;
	}
	else if(t==2)//East
	{
		if((im.data[(i-1)*cols+(j-1)]==(uchar)255)&&(im.data[(i-0)*cols+(j-1)]==(uchar)255)&&(im.data[(i+1)*cols+(j-1)]==(uchar)255)&&(im.data[(i-1)*cols+(j+1)]==0)&&(im.data[(i-0)*cols+(j+1)]==0)&&(im.data[(i+1)*cols+(j+1)]==0))
			return true;
		else
			return false;
	}
	else if(t==3)//South
	{
		if((im.data[(i+1)*cols+(j-1)]==(uchar)0)&&(im.data[(i+1)*cols+(j-0)]==(uchar)0)&&(im.data[(i+1)*cols+(j+1)]==(uchar)0)&&(im.data[(i-1)*cols+(j-1)]==(uchar)255)&&(im.data[(i-1)*cols+(j-0)]==(uchar)255)&&(im.data[(i-1)*cols+(j+1)]==(uchar)255))
			return true;
		else
			return false;
	}
	else//West
	{
		if((im.data[(i-1)*cols+(j-1)]==(uchar)0)&&(im.data[(i-0)*cols+(j-1)]==(uchar)0)&&(im.data[(i+1)*cols+(j-1)]==(uchar)0)&&(im.data[(i-1)*cols+(j+1)]==(uchar)255)&&(im.data[(i-0)*cols+(j+1)]==(uchar)255)&&(im.data[(i+1)*cols+(j+1)]==(uchar)255))
			return true;
		else
			return false;
	}
}
//Checks for matching corner templates
bool Processor::cornerMatch(Mat im, int i, int j, int t){
		int cols=im.cols;
		if(t==1)//NE
		{
			if((im.data[i*cols+(j-1)==(uchar)255])&&(im.data[(i+1)*cols+(j-0)==(uchar)255])&&(im.data[(i-1)*cols+(j-0)==0])&&(im.data[(i-1)*cols+(j+1)==0])&&(im.data[i*cols+(j+1)==0]))
				return true;
			else
				return false;
		}
		else if(t==2)//SE
		{
			if((im.data[i*cols+(j-1)==(uchar)255])&&(im.data[(i-1)*cols+(j-0)==(uchar)255])&&(im.data[(i+1)*cols+(j-0)==0])&&(im.data[(i+1)*cols+(j+1)==0])&&(im.data[i*cols+(j+1)==0]))
				return true;
			else
				return false;
		}
		else if(t==3)//SW
		{
			if((im.data[i*cols+(j+1)==(uchar)255])&&(im.data[(i-1)*cols+(j-0)==(uchar)255])&&(im.data[(i+1)*cols+(j-0)==0])&&(im.data[(i+1)*cols+(j-1)==0])&&(im.data[i*cols+(j-1)==0]))
				return true;
			else
				return false;
		}
		else//NW
		{
			if((im.data[i*cols+(j+1)==(uchar)255])&&(im.data[(i+1)*cols+(j-0)==(uchar)255])&&(im.data[(i-1)*cols+(j-0)==0])&&(im.data[(i-1)*cols+(j-1)==0])&&(im.data[i*cols+(j-1)==0]))
				return true;
			else
				return false;
		}
}
//Calculates Connectivity Number
int Processor::connectivityNumber(Mat im, int i, int j){
	int cols=im.cols;
	int N1,N2,N3,N4,N5,N6,N7,N8;
	N1=(int)(im.data[i*cols+(j+1)])/255;
	N2=(int)(im.data[(i+1)*cols+(j+1)])/255;
	N3=(int)(im.data[(i+1)*cols+(j)])/255;
	N4=(int)(im.data[(i+1)*cols+(j-1)])/255;
	N5=(int)(im.data[i*cols+(j-1)])/255;
	N6=(int)(im.data[(i-1)*cols+(j-1)])/255;
	N7=(int)(im.data[(i-1)*cols+(j)])/255;
	N8=(int)(im.data[(i-1)*cols+(j+1)])/255;
	int sum1,sum2,sum3,sum4;
	sum1=N1-(N1*N2*N3);
	sum2=N3-(N3*N4*N5);
	sum3=N5-(N5*N6*N7);
	sum4=N7-(N7*N8*N1);
	return (sum1+sum2+sum3+sum4);
}

//Calculates Number of Neighbors for a pixel (i,j)
int Processor::neighbors(Mat im, int i, int j) {
int cols = im.cols;

	int n=0;
	if(im.data[i*cols+(j+1)]==(uchar)255)
		n++;
	if(im.data[(i+1)*cols+(j+1)]==(uchar)255)
		n++;
	if(im.data[(i+1)*cols+(j-0)]==(uchar)255)
		n++;
	if(im.data[(i+1)*cols+(j-1)]==(uchar)255)
		n++;
	if(im.data[i*cols+(j-1)]==(uchar)255)
		n++;
	if(im.data[(i-1)*cols+(j-1)]==(uchar)255)
		n++;
	if(im.data[(i-1)*cols+(j-0)]==(uchar)255)
		n++;
	if(im.data[(i-1)*cols+(j+1)]==(uchar)255)
		n++;

	return n;
}
//Calculates the S-(transition) value
int Processor::svalue(Mat im, int i, int j){
	int s=0;
	int cols=im.cols;
	//Count the transitions
	if(im.data[(i-1)*cols+j]==0&&im.data[(i-1)*cols+(j+1)]==(uchar)255)
		s++;
	if(im.data[(i-1)*cols+(j+1)]==0&&im.data[(i-0)*cols+(j+1)]==(uchar)255)
		s++;
	if(im.data[(i-0)*cols+(j+1)]==0&&im.data[(i+1)*cols+(j+1)]==(uchar)255)
		s++;
	if(im.data[(i+1)*cols+(j+1)]==0&&im.data[(i+1)*cols+(j-0)]==(uchar)255)
		s++;
	if(im.data[(i+1)*cols+(j-0)]==0&&im.data[(i+1)*cols+(j-1)]==(uchar)255)
		s++;
	if(im.data[(i+1)*cols+(j-1)]==0&&im.data[(i-0)*cols+(j-1)]==(uchar)255)
		s++;
	if(im.data[(i-0)*cols+(j-1)]==0&&im.data[(i-1)*cols+(j-1)]==(uchar)255)
		s++;
	if(im.data[(i-1)*cols+(j-1)]==0&&im.data[(i-1)*cols+(j-0)]==(uchar)255)
		s++;
	return s;
}
//Calculates Number of 4-Connected Neighbors for a pixel (i,j)
int Processor::neighbors4(Mat im, int i, int j) {
	//Take the maximum of 4 cross and 4 diagonal neighbors
	int cols = im.cols;

	int n1=0,n2=0;
	if(im.data[i*cols+(j+1)]==(uchar)255)
		n1++;
	if(im.data[(i+1)*cols+(j+1)]==(uchar)255)
		n2++;
	if(im.data[(i+1)*cols+(j-0)]==(uchar)255)
		n1++;
	if(im.data[(i+1)*cols+(j-1)]==(uchar)255)
		n2++;
	if(im.data[i*cols+(j-1)]==(uchar)255)
		n1++;
	if(im.data[(i-1)*cols+(j-1)]==(uchar)255)
		n2++;
	if(im.data[(i-1)*cols+(j-0)]==(uchar)255)
		n1++;
	if(im.data[(i-1)*cols+(j+1)]==(uchar)255)
		n2++;

	return max(n1,n2);
}
//Detects the most likely start and end regions using a bounding box
Mat Processor::detectStartEndRegions(Mat im){
	//Assume image is binary
	int cols=im.cols;
	int rows=im.rows;

	/*
	//Calculate x,y (via min height and width of the solution space)
	int minw=cols+1,minh=rows+1;
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++)
			if(im.data[i*cols+j]==(uchar)255){
				if(i<minh)
					minh=i;
				if(j<minw)
					minw=j;
			}
	//Calculate width and height
	int maxw=-1,maxh=-1;
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++)
			if(im.data[i*cols+j]==(uchar)255){
				if(i>maxh)
					maxh=i;
				if(j>maxw)
					maxw=j;
			}
	*/
	//Calculates maxh,minh,maxw,minw using the contours of the maze

	int x,y;
	int minw=cols+1,minh=rows+1,maxw=-1,maxh=-1;
	for(int i=0;i<glocombcont.size();i++){
		x=glocombcont[i].x;
		y=glocombcont[i].y;
		if(y<minh)
			minh=y;
		if(x<minw)
			minw=x;
		if(y>maxh)
			maxh=y;
		if(x>maxw)
			maxw=x;
	}
	int mpw=5;//Minimum pixel width (original: mpw=20)
	Mat mask = cvCreateMat(rows, cols, CV_8UC1);
	mask.setTo(cv::Scalar(0));
	for(int i=minh;i<maxh;i++)
		for(int j=minw;j<maxw;j++)
			if(i<(minh+mpw)||j<(minw+mpw)||i>(maxh-(mpw+1))||j>(maxw-(mpw+1)))
				mask.data[i*cols+j]=(uchar)255;
	bitwise_and(mask,im,im);
	//im=erodeImage(im,5,1);
	__android_log_print(ANDROID_LOG_VERBOSE,"yxhw" ,"%d, %d, %d, %d",minh,minw,maxh,maxw);
	return im;
}


