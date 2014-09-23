/*************************************************************************************************************************
 Copyright GEORGAKIS GEORGIOS, CHRISTODOULOU SAVVAS (c) 2013, ggeorgak@masonlive.gmu.edu, savvaschris@yahoo.com  

 Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted,
 provided that the above copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE 
 AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, 
 NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

*************************************************************************************************************************/

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace std; 

float AverageIntensity(Mat image);
bool CheckWider(Mat img, int value, int i, int j);
bool CheckMaxima(Mat scale_Norm[], int scale_levels, int value, int i, int j);
float FindMaxima(int scale_levels, Mat scale_Norm[], double sigmas[]);
vector<float> BlobDetection(Mat src_image);
vector<float> FindFeatures(Mat img);
vector<float> SiftFeatures(Mat image);
vector<float> AverageColours(Mat image);
float StandardDeviation(vector<float> population, float N);
vector<float> colorHistograms(Mat img);
float findLines(Mat img);
vector<float> GradientOrientation(Mat image);
float EuclideanDistance(vector<float> vec1, vector<float> vec2);
vector<int> FindNearestImages(Mat trainData, Mat trainClasses, Mat testData, int featNum, vector<int> ImageNames);
float CosineSimilarity(vector<float> vec1, vector<float> vec2);
vector<float> ContourFeatures(Mat image);
void DisplayImages(vector<int> filenames);

vector<float> x, y, radius; 

int main( int argc, char** argv ){

	Mat test_image = imread("97.jpg");
	
	clock_t startTime = clock();

	int trainCounter=0;
	int trainingNumber = 800, testNumber = 200, featureNumber = 20;
	Mat trainData (trainingNumber, featureNumber, CV_32FC1);
	Mat trainClasses (trainingNumber, 1, CV_32FC1);
	Mat testSample(1, featureNumber, CV_32FC1);
	vector<int> ImageNames;

	/*Collect the training data*/
	for (int Class=0; Class<=9; Class++){
		for (int img=0; img<=79; img++){
			int image_number = img + Class*100;
			String number;
			ostringstream conv_n;
			conv_n << image_number;
			number = conv_n.str();
			Mat src_color = imread("image.orig/"+number+".jpg");
			ImageNames.push_back(image_number);

			/*Extract the features of the source image*/
			vector<float> feat = FindFeatures(src_color);

			/*Label the classes and fill the training data*/
			trainClasses.at<float>(trainCounter) = Class;

			for (int w=0; w<trainData.cols; w++){
				trainData.at<float>(trainCounter, w) = feat.at(w);
			}
			trainCounter++;
		}
	}

	cout << endl;

	/*Find features of test image*/
	vector<float> test_feat = FindFeatures(test_image);
	for (int w=0; w<testSample.cols; w++)
		testSample.at<float>(0, w) = test_feat.at(w);

	vector<int> files = FindNearestImages(trainData, trainClasses, testSample, featureNumber, ImageNames);

	DisplayImages(files);

	cout << endl;
	cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << endl;
	
	/*Wait for a keystroke in the window*/
	imshow("Test Image", test_image);
    waitKey(0);                                          
    return 0;
}

/*Display the 20 nearest images of the test image*/
void DisplayImages(vector<int> filenames){

	for (int i=0; i<filenames.size(); i++){
		int file = filenames.at(i);
		String number_file;
		ostringstream conv_file;
		conv_file << file;
		number_file = conv_file.str();
		Mat file_color = imread("image.orig/"+number_file+".jpg");
		imshow("Result "+number_file, file_color);
	}

}

/*Calculate difference between two vectors using euclidean distance or average distance*/
float EuclideanDistance(vector<float> vec1, vector<float> vec2){
	float euclid=0, average=0;
	vector<float> diff;
	for (int i=0; i<vec1.size(); i++){
		float value1 = vec1.at(i);
		float value2 = vec2.at(i);

		euclid = euclid + pow((value1 - value2), 2);

		average = average + abs(value1-value2);
		diff.push_back(abs(value1-value2));
	}

	float std = StandardDeviation(diff, diff.size());
	
	//return sqrt(euclid);
	return average/vec1.size();
	//return std;
}

/*Calculate difference between two vectors using cosine similarity*/
float CosineSimilarity(vector<float> vec1, vector<float> vec2){
	float cosine=0, sumAB=0, sumA=0, sumB=0;

	for (int i=0; i<vec1.size(); i++){
		float value1 = vec1.at(i);
		float value2 = vec2.at(i);

		sumAB = sumAB + (value1*value2);
		sumA = sumA + pow(value1, 2);
		sumB = sumB + pow(value2, 2);
	}

	float par1 = sqrt(sumA);
	float par2 = sqrt(sumB);

	cosine = sumAB/(par1*par2);

	return cosine;
}

/*Find the images that are most similar to the test image*/
vector<int> FindNearestImages(Mat trainData, Mat trainClasses, Mat testData, int featNum, vector<int> ImageNames){
	vector<float> trainImage, testImage, distances;
	vector<int> files;
	int numOfResults = 20;

		/*Get feature vector of current test image*/
		for (int i=0; i<featNum; i++){
			testImage.push_back(testData.at<float>(0, i));
		}

		/*Get one by one the feature vectors of train images and compare with current test image*/
		for (int i=0; i<trainData.rows; i++){

			/*Get feature vector of the specific train image*/
			for (int j=0; j<featNum; j++){
				trainImage.push_back(trainData.at<float>(i, j));
			}

			/*Calculate Euclidean distance between test image and train image*/
			float dist = EuclideanDistance(trainImage, testImage);
			//float dist = CosineSimilarity(trainImage, testImage);
			distances.push_back(dist);
		
			//cout << "Image:" << ImageNames.at(i) << "  Class:" << trainClasses.at<float>(i) << "  Dist:" << dist << endl;

			trainImage.clear();
		}
		testImage.clear();

		int numOfCorrect = 0;

		/*Find 20 nearest images using the distances*/
		for (int n=0; n<numOfResults; n++){
			int filename=0, Class=0, index=0;
			float min = distances.at(0);
			for (int i=1; i<distances.size(); i++){
				if (min>distances.at(i)){
					min = distances.at(i);
					filename = ImageNames.at(i);
					Class = trainClasses.at<float>(i);
					index = i;
				}
			}
			distances.at(index) = 9999;
			
			cout << "Image:" << filename << "  Class:" << Class << "  Dist:" << min << endl;
			files.push_back(filename);
		}
		distances.clear();

		return files;
}

/*Return the feature vector of an image*/
vector<float> FindFeatures(Mat img){
	vector<float> features;

	/*Blob features (x, y, radius, numberOf)*/
	vector<float> blob_f = BlobDetection(img);
	for (int i=0; i<blob_f.size(); i++){
		features.push_back(blob_f.at(i));
	}

	/*Surf features (x, y, angle, numberOf)*/
	vector<float> sift_f = SiftFeatures(img);
	for (int i=0; i<sift_f.size(); i++){
		features.push_back(sift_f.at(i));
	}

	/*Color features (red, green, blue)*/
	vector<float> color_feat = AverageColours(img);
	for (int i=0; i<color_feat.size(); i++){
		features.push_back(color_feat.at(i));
	}

	/*Average Intensity feature*/
	float I_average = AverageIntensity(img);
	features.push_back(I_average);

	/*Color histogram features (red, green, blue)*/
	vector<float> hist_feat = colorHistograms(img);
	for (int i=0; i<hist_feat.size(); i++){
		features.push_back(hist_feat.at(i));
	}

	/*Hough transform number of lines features (num of lines)*/
	float hough = findLines(img);
	features.push_back(hough);

	/*Gradient orientation features (orientation, magnitude)*/
	vector<float> grad_f = GradientOrientation(img);
	for (int i=0; i<grad_f.size(); i++){
		features.push_back(grad_f.at(i));
	}

	/*Find Contour features (average size, number of)*/
	vector<float> contour_f = ContourFeatures(img);
	for (int i=0; i<contour_f.size(); i++){
		features.push_back(contour_f.at(i));
	}

	return features;
}

/*Calculate features from SIFT*/
vector<float> SiftFeatures(Mat image){
	Mat gray;
	cvtColor(image, gray, CV_RGB2GRAY);
	vector<float> sift_x, sift_y, sift_angle, sift_features;
	float stdX=0, stdY=0, stdA=0;

	SiftFeatureDetector detector;
	//SurfFeatureDetector detector(600);
	vector<KeyPoint> keypoints;
	detector.detect(gray, keypoints);
	
	if (keypoints.size()!=0){
		for (int i=0; i<keypoints.size(); i++){
			sift_x.push_back(keypoints.at(i).pt.x);
			sift_y.push_back(keypoints.at(i).pt.y);
			sift_angle.push_back(keypoints.at(i).angle);
		}

		stdX = StandardDeviation(sift_x, keypoints.size());
		stdY = StandardDeviation(sift_y, keypoints.size());
		stdA = StandardDeviation(sift_angle, keypoints.size()); 
	}

	//cout << "X:" << stdX << "  Y:" << stdY << "  Angle:" << stdA << "  Number:" << keypoints.size() << endl;

	sift_features.push_back(stdX);
	sift_features.push_back(stdY);
	sift_features.push_back(stdA);
	sift_features.push_back(keypoints.size());

	return sift_features;
}

/*Calculate standard deviation for a population of numbers*/
float StandardDeviation(vector<float> population, float N){
	float sum=0, av=0, dev_sum=0, std=0;
	if (N!=0){
		
		for (int i=0; i<N; i++)
			sum = sum + population.at(i);

		av = sum/N;

		for (int i=0; i<N; i++)
			dev_sum = dev_sum + pow(population.at(i)-av, 2);

		std = sqrt(dev_sum/N);
	}

	return std;
}

/*Calculate features from Blob detection*/
vector<float> BlobDetection(Mat src_image){
	vector<float> blobs;
	const int scale_levels = 7;
	Mat gray_image; 
	Mat	scale_Gaussian[scale_levels];
	Mat scale_Laplacian[scale_levels];
	Mat scale_Norm[scale_levels];
	double sigmas[scale_levels];
	int kernel_size = 3, scale = 1, delta = 0, ddepth = CV_8UC1;

	/*Convert to grayscale*/
	cvtColor(src_image, gray_image, CV_RGB2GRAY);

	/*Create scale space*/ 
	for (int i=0; i<scale_levels; i++){

		double sigma_gaussian = (i+1)*2;
		//double sigma_gaussian = (i+1)*2;
		int kernel_gaussian = ((i+1) * 4) + 1;
		//int kernel_gaussian = sigma_gaussian*3;
		//if (kernel_gaussian%2==0) kernel_gaussian++;

		GaussianBlur(gray_image, scale_Gaussian[i], Size(kernel_gaussian,kernel_gaussian), sigma_gaussian);

		Laplacian(scale_Gaussian[i], scale_Laplacian[i], ddepth, kernel_size, scale, delta);

		scale_Norm[i] = scale_Laplacian[i] * sigma_gaussian^2;
		//scale_Norm[i] = scale_Norm[i] * 2;
		sigmas[i] = sigma_gaussian;
	
	}

	/*Iterate through all scale spaces and find blob positions*/
	float maxima_number = FindMaxima(scale_levels, scale_Norm, sigmas);

	/*Calculate standard deviation of coordinates of blobs and radius for features*/
	float stdX = StandardDeviation(x, maxima_number);
	float stdY = StandardDeviation(y, maxima_number);
	float stdR = StandardDeviation(radius, maxima_number);
	
	blobs.push_back(stdX);
	blobs.push_back(stdY);
	blobs.push_back(stdR);
	blobs.push_back(maxima_number);

	//cout << "Counter:" << maxima_number << "  SumX:" << sum_x << endl;
	//cout << "X:" << stdX << "  Y:" << stdY << "  R:" << stdR << "  Number of:" << maxima_number << endl;
	
	x.clear();
	y.clear();
	radius.clear();

	return blobs;
}
float FindMaxima(int scale_levels, Mat scale_Norm[], double sigmas[]){
	float maxima_counter = 0;
	int threshold = 125;
	for (int s=1; s<scale_levels-1; s++){
		/*Get space scale and sigma*/
		cv::Mat img = scale_Norm[s];
		double sigma = sigmas[s];
		
		/*Iterate through pixels*/
		for (int i=2; i<img.rows-2; i++){
			for (int j=2; j<img.cols-2; j++){

				int value = img.at<uchar>(i,j);

				/*If the intensity value is lower than the theshold, then the pixel is not a maxima*/
				if (value < threshold) continue;

				/*Check if a pixel is a maximum in all space scales*/
				if (CheckMaxima(scale_Norm, scale_levels, value, i, j)) continue;

				/*Check if a pixel is the maximum from its neighbors in its space scale*/
				if ( img.at<uchar>(i-1,j) >= value || img.at<uchar>(i+1,j) >= value || img.at<uchar>(i,j-1) >= value || 
					img.at<uchar>(i,j+1) >= value || img.at<uchar>(i-1,j-1) >= value || img.at<uchar>(i+1,j-1) >= value || 
					img.at<uchar>(i-1,j+1) >= value || img.at<uchar>(i+1,j+1) >= value ){
					continue;
				}

				/*Check wider in the neighborhood*/
				if (CheckWider(img, value, i, j)) continue;

				/*Check if a pixel is the maximum from its neighbors in the previous space scale*/
				cv::Mat scalePrevious = scale_Norm[s-1];
				if ( scalePrevious.at<uchar>(i-1,j) >= value || scalePrevious.at<uchar>(i+1,j) >= value || 
					scalePrevious.at<uchar>(i,j-1) >= value || scalePrevious.at<uchar>(i,j+1) >= value || 
					scalePrevious.at<uchar>(i-1,j-1) >= value || scalePrevious.at<uchar>(i+1,j-1) >= value || 
					scalePrevious.at<uchar>(i-1,j+1) >= value || scalePrevious.at<uchar>(i+1,j+1) >= value ||
					scalePrevious.at<uchar>(i,j) >= value){
					continue;
				}

				/*Check wider in the neighborhood*/
				if (CheckWider(scalePrevious, value, i, j)) continue;

				/*Check if a pixel is the maximum from its neighbors in the next space scale*/
				cv::Mat scaleNext = scale_Norm[s+1];
				if ( scaleNext.at<uchar>(i-1,j) >= value || scaleNext.at<uchar>(i+1,j) >= value || 
					scaleNext.at<uchar>(i,j-1) >= value || scaleNext.at<uchar>(i,j+1) >= value || 
					scaleNext.at<uchar>(i-1,j-1) >= value || scaleNext.at<uchar>(i+1,j-1) >= value || 
					scaleNext.at<uchar>(i-1,j+1) >= value || scaleNext.at<uchar>(i+1,j+1) >= value ||
					scaleNext.at<uchar>(i,j) >= value){
					continue;
				}

				/*Check wider in the neighborhood*/
				if (CheckWider(scaleNext, value, i, j)) continue;
			
				/*Calculate radius of blob and store the coordinates and the radius*/
				double rad = sigma*1.414;
				//cout << "Found Maxima at:(" << i << "," << j << ") Sigma:" << sigma << " Radius:" << rad << " Intensity:" << value << endl;
				x.push_back(i);
				y.push_back(j);
				radius.push_back(rad);
				maxima_counter++;
			}
		}
	}
	return maxima_counter;
}
bool CheckMaxima(Mat scale_Norm[], int scale_levels, int value, int i, int j){
	bool notMaxima = false; 
	for (int scale=0; scale<scale_levels; scale++){
		Mat sc = scale_Norm[scale];
		/*If the pixel has lower intensity then it is not a maximum*/
		if (value < sc.at<uchar>(i,j)){
			notMaxima = true;
			break;
		}
	}

	return notMaxima;
}
bool CheckWider(Mat img, int value, int i, int j){
	if ( img.at<uchar>(i-2,j-2) >= value || img.at<uchar>(i-2,j-1) >= value || img.at<uchar>(i-2,j) >= value || 
		img.at<uchar>(i-2,j+1) >= value || img.at<uchar>(i-2,j+2) >= value || img.at<uchar>(i-1,j+2) >= value || 
		img.at<uchar>(i,j+2) >= value || img.at<uchar>(i+1,j+2) >= value || img.at<uchar>(i+2,j+2) >= value || 
		img.at<uchar>(i+2,j+1) >= value || img.at<uchar>(i+2,j) >= value || img.at<uchar>(i+2,j-1) >= value ||
		img.at<uchar>(i+2,j-2) >= value || img.at<uchar>(i+1,j-2) >= value || img.at<uchar>(i,j-2) >= value ||
		img.at<uchar>(i-1,j-2) >= value ){
			return true;
		}

	return false;
}

/*Calculate average intensity of the image as a feature*/
float AverageIntensity(Mat image){
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);

	float intensity=0, average=0, pixel_num;
	for (int i=0; i<gray.rows; i++){
		for (int j=0; j<gray.cols; j++){
			intensity = intensity + gray.at<uchar>(i,j);
		}
	}
	pixel_num = gray.rows * gray.cols;
	average = intensity / pixel_num;

	return average;
}

/*Calculate average value of each color channel*/
vector<float> AverageColours(Mat image){
	float sum_red=0, sum_green=0, sum_blue=0;
	vector<float> colors; 
	//vector<float> Blue, Green, Red;

	for (int i=0; i<image.rows; i++){
		for (int j=0; j<image.cols; j++){
			int blue = (int)image.at<Vec3b>(i,j)[0];
			int green = (int)image.at<Vec3b>(i,j)[1];
			int red = (int)image.at<Vec3b>(i,j)[2];

			sum_blue = sum_blue + blue;
			sum_green = sum_green + green;
			sum_red = sum_red + red;
			/*Blue.push_back(blue);
			Green.push_back(green);
			Red.push_back(red);*/
		}
	}

	int numberOf = image.rows*image.cols;
	colors.push_back(sum_blue/numberOf);
	colors.push_back(sum_green/numberOf);
	colors.push_back(sum_red/numberOf);

	//float stdB = StandardDeviation(Blue, Blue.size());
	//float stdG = StandardDeviation(Green, Green.size());
	//float stdR = StandardDeviation(Red, Red.size());
	//colors.push_back(stdB);
	//colors.push_back(stdG);
	//colors.push_back(stdR);

	return colors;
}

/*Find max color values using histograms*/
vector<float> colorHistograms(Mat img)
{
	vector<float> result;

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( img, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	//for( int i = 1; i < histSize; i++ )
	//{
	//		cout<<i<<": "<<r_hist.at<float>(i)<<" "<<g_hist.at<float>(i)<<" "<<b_hist.at<float>(i)<<endl;  
	//}

	int maxR=0, maxG=0, maxB=0, maxRi=0, maxGi=0, maxBi=0;
    for( int i = 1; i < histSize; i++ )
	{
			if(r_hist.at<float>(i)>maxR)
			{
				maxR = r_hist.at<float>(i);
				maxRi=i;
			}
			if(g_hist.at<float>(i)>maxG)
			{
				maxG = g_hist.at<float>(i);
				maxGi=i;
			}
			if(b_hist.at<float>(i)>maxB)
			{
				maxB = b_hist.at<float>(i);
				maxBi=i;
			}
	}

	result.push_back(maxRi);
	result.push_back(maxGi);
	result.push_back(maxBi);

	return result;
}

/*Find number of lines in the image using Hough*/
float findLines(Mat img)
{

  Mat dst, gblur;
  GaussianBlur(img, gblur, Size(7,7), 7/3);
  Canny(gblur, dst, 60, 60, 3);
  //cvtColor(dst, cdst, CV_GRAY2BGR);

  vector<Vec2f> lines;
  HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

  //vector<float> numOfLines = lines.size();
  //cout << (float)lines.size() << endl;

  return (float)lines.size();
}

/*Calculate the average orientation and magnitude of the image*/
vector<float> GradientOrientation(Mat image){
	int ddepth = CV_8UC1;
	vector<float> orientations, magnitudes, grad_feat;
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(7,7), 7/3);

	Mat grad_x, grad_y;
	Sobel(gray, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(gray, grad_y, ddepth, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	float rx=0, ry=0, orient=0, mag=0, sum_orient=0, sum_mag=0;
	for (int i=0; i<grad_x.rows; i++){
		for (int j=0; j<grad_x.cols; j++){
			rx = (float)grad_x.at<uchar>(i,j);
			ry = (float)grad_y.at<uchar>(i,j);
			
			orient = atan2(ry, rx);
			mag = sqrt(pow(rx,2) + pow(ry,2));
			//cout << "Orient:" << orient << "  Mag:" << mag << endl;
			orientations.push_back(orient);
			magnitudes.push_back(mag);		
		}
	}

	float stdOrient = StandardDeviation(orientations, orientations.size());
	float stdMag = StandardDeviation(magnitudes, magnitudes.size());

	grad_feat.push_back(stdOrient);
	grad_feat.push_back(stdMag);

	return grad_feat;
}

/*Calculate the average size and number of contours in the image*/
vector<float> ContourFeatures(Mat image){
	vector<float> contours_feat;
	Mat canny_output;
	vector<vector<Point>> contours;
	Mat gray, blur;
	cvtColor(image, gray, CV_BGR2GRAY);
	GaussianBlur(gray, blur, Size(7,7), 7/3);

	// Detect edges using canny
	Canny(blur, canny_output, 180, 220, 3);
	// Find contours
	findContours(canny_output, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	//imshow("Canny", canny_output);

	/*Find average size of the contours in the image*/
	float average=0;
	if (contours.size()!=0){
		float sum=0;
		for (int i=0; i<contours.size(); i++){
			sum = sum + contours.at(i).size();
		}

		average = sum/contours.size();
	}

	contours_feat.push_back(average);
	contours_feat.push_back(contours.size());

	return contours_feat;
}