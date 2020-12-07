#include <stdio.h>
#include <iostream>
#include <filesystem>

#include <iostream>
#include <fstream>
#include <sstream>

#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

#include "opencv2/dnn.hpp"

#include "SerialPort.hpp"

#include "UntrainedDetector.hpp"
#include "TrainedDetector.hpp"


#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME	"OpenCV and Arduino"

#include <iostream>
#include <memory>

#define DATA_LENGTH 255

const char* portName = "\\\\.\\COM4";

//Declare a global object
std::shared_ptr<SerialPort> arduino;

char receivedString[DATA_LENGTH];

bool firstRun = true;

bool ledOn = false;

bool untrainedRecognition = true;

void ledTestDraw(cv::Mat& frame) {

	cvui::text(frame, 40, 40, "Click para comunicarse con Arduino");

	if (cvui::button(frame, 300, 80, "Encender")) {
		const char* sendString = "ON\n";
		bool hasWritten = arduino->writeSerialPort(sendString, DATA_LENGTH);
		if (hasWritten) std::cout << "Datos escritos correctamente" << std::endl;
	else std::cerr << "Datos no escritos" << std::endl;
	}
	

	if (cvui::button(frame, 300, 140, "Apagar")) {
		const char* sendString = "OFF\n";
		bool hasWritten = arduino->writeSerialPort(sendString, DATA_LENGTH);
	if (hasWritten) std::cout << "Datos escritos correctamente" << std::endl;
	else std::cerr << "Datos no escritos" << std::endl;
	}

	int hasRead = arduino->readSerialPort(receivedString, DATA_LENGTH);
	if (hasRead)
	{
		std::cout << receivedString << "\n";


		if (strcmp(receivedString, "ON") == 0) {
			ledOn = true;
		}

		if (strcmp(receivedString, "OFF") == 0) {
			ledOn = false;
		}
	}



	if (ledOn) {
		cvui::text(frame, 40, 120, "Led 13 encendido");
	}
	else {
		cvui::text(frame, 40, 120, "Led 13 apagado");
	}

}



int main(int argc, char* argv[])
{
	arduino = std::make_shared<SerialPort>(portName);
	std::cout << "¿Conectado? : " << arduino->isConnected() << std::endl;

	if (!arduino->isConnected()) {
		std::cout << "No se pudo conectar a Arduino" << std::endl;
		//return -1;
	}

	cvui::init(WINDOW_NAME, 20);

	cv::VideoCapture cap(0); // open the video capture for reading
	if (!cap.isOpened()) // if not success, exit program
	{
		std::cout << "No se pudo abrir la cámara" << std::endl;
		return -1;
	}

	cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

	std::cout << "Directorio actual: " << std::filesystem::current_path() << '\n';

	TrainedDetector trained_detector;
	UntrainedDetector untrained_detector;

	auto current_path = std::filesystem::current_path();
	std::string jennette_path = "data/images/jennette_test.jpg";

	std::string jennette_full_path = current_path.append(jennette_path).string();

	cv::Mat jennette_test = cv::imread(jennette_full_path);

	while (true) {
		cv::Mat frame;

		bool bSuccess = cap.read(frame);

		//ledTestDraw(frame);

		int rows = std::max(frame.rows, jennette_test.rows);
		int cols = frame.cols + jennette_test.cols;

		// Create a black image
		cv::Mat res(rows, cols, CV_8UC3);

		frame.copyTo(res(cv::Rect(0, 0, frame.cols, frame.rows)));

		jennette_test.copyTo(res(cv::Rect(0, 0, jennette_test.cols, jennette_test.rows)));


		frame = res;

		cv::Mat detection;
		
		if (untrainedRecognition) {
			if (bSuccess) {
				detection = untrained_detector.detect(frame);
			}
			else {
				detection = frame;
			}

		}
		else {
			auto rectangles = trained_detector.detect_face_rectangles(frame);
			cv::Scalar color(0, 105, 205);
			int frame_thickness = 4;
			for (const auto& r : rectangles) {
				cv::rectangle(frame, r, color, frame_thickness);
			}

			detection = frame;
		}

		cvui::update();

		cv::imshow(WINDOW_NAME, detection);
	}

	return 0;
}
