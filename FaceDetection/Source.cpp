#include <iostream>
#include <string>

#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat drawFaces(cv::Mat frame, std::vector<cv::Rect> faces);
std::vector<cv::Rect> detectFace(cv::Mat frame,cv::CascadeClassifier faceCascade);
void detectAndShow(cv::Mat frame,cv::CascadeClassifier faceCascade,cv::CascadeClassifier eyesCascade);

int main(){
	setlocale(LC_ALL,"");
	cv::VideoCapture capture(0);
	cv::CascadeClassifier matchTemplate;

	try{
		if(!capture.isOpened())
			throw (std::string("Error opening video stream"));
		if(!matchTemplate.load("resources/haarcascade_frontalface_default.xml"))
			throw (std::string("Error opening template"));
	}catch(std::string error){
		std::cout<<error<<std::endl;
		capture.release();
		return 0;
	}

	cv::Mat frame;
	cv::Mat grayscale;
	while(capture.isOpened()){
		capture>>frame;

		cv::cvtColor(frame,grayscale,cv::COLOR_BGR2GRAY);
		cv::equalizeHist(grayscale,grayscale);

		std::vector<cv::Rect> faces = detectFace(grayscale,matchTemplate);
		frame = drawFaces(frame,faces);
		imshow("Camera", frame);

		char key = cv::waitKey(11);
		if(key == 27) break;
	}
}

std::vector<cv::Rect> detectFace(cv::Mat frame,cv::CascadeClassifier faceCascade){
	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(frame,faces, 1.1, 2,0,cv::Size(30,30));
	return faces;
}

cv::Mat drawFaces(cv::Mat frame, std::vector<cv::Rect> faces){
	cv::Mat result = frame.clone();

	for(size_t i = 0;i<faces.size();i++){
		cv::Point firstPoint(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		cv::Point secondPoint(faces[i].x, faces[i].y);
		cv::rectangle(result,firstPoint, secondPoint,cvScalar(0,255,0), 1, 8,0);
	}

	return result;
}