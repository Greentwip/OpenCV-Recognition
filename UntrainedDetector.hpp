#pragma once

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <ostream>
#include <istream>


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

std::string g_listname_t[] =
{
	"Victor Lopez",
	"Jennette McCurdy"
};


class UntrainedDetector {

public:
	UntrainedDetector() {
		auto current_path = std::filesystem::current_path();
		std::string face_cascade_path = "data/haarcascades/haarcascade_frontalface_default.xml";

		std::string face_full_path = current_path.append(face_cascade_path).string();

		_haar_cascade.load(face_full_path);

		std::string fn_csv = "data/csv.ext";

		try {
			read_csv(fn_csv, _images, _labels);
		}
		catch (cv::Exception& e) {
			std::cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << std::endl;
			exit(1);
		}

		_im_width = _images[0].cols;
		_im_height = _images[0].rows;

		_model = cv::face::EigenFaceRecognizer::create();

		_model->train(_images, _labels);

	}

	static void read_csv(const std::string& filename,
		std::vector<cv::Mat>& images,
		std::vector<int>& labels,
		char separator = ';') {
		std::ifstream file(filename.c_str(), std::ifstream::in);
		if (!file) {
			std::string error_message = "No valid input file was given, please check the given filename.";
			CV_Error(cv::Error::StsBadArg, error_message);
		}
		std::string line, path, classlabel;
		while (getline(file, line)) {
			std::stringstream liness(line);
			getline(liness, path, separator);
			getline(liness, classlabel);
			if (!path.empty() && !classlabel.empty()) {

				auto current_path = std::filesystem::current_path();
				std::string image_path = path;

				std::string image_full_path = current_path.append(image_path).string();

				images.push_back(cv::imread(image_full_path, 0));
				labels.push_back(atoi(classlabel.c_str()));
			}
		}
	}

	cv::Mat detect(cv::Mat& frame) {
		cv::Mat original = frame.clone();
		// Convert the current frame to grayscale:
		cv::Mat gray;
		cvtColor(original, gray, cv::COLOR_BGR2GRAY);
		// Find the faces in the frame:
		std::vector< cv::Rect_<int> > faces;
		_haar_cascade.detectMultiScale(gray, faces);
		// At this point you have the position of the faces in
		// faces. Now we'll get the faces, make a prediction and
		// annotate it in the video. Cool or what?
		for (int i = 0; i < faces.size(); i++) {
			// Process face by face:
			cv::Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			cv::Mat face = gray(face_i);
			// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
			// verify this, by reading through the face recognition tutorial coming with OpenCV.
			// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
			// input data really depends on the algorithm used.
			//
			// I strongly encourage you to play around with the algorithms. See which work best
			// in your scenario, LBPH should always be a contender for robust face recognition.
			//
			// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
			// face you have just found:
			cv::Mat face_resized;
			cv::resize(face, face_resized, cv::Size(_im_width, _im_height), 1.0, 1.0, cv::INTER_CUBIC);
			// Now perform the prediction, see how easy that is:

			double confidence = 0.0;
	
			int prediction = 0;
			_model->predict(face_resized, prediction, confidence);

			if (confidence < 11000) {
				prediction = 0;
			}
			else {
				prediction = 1;
			}

			// And finally write all we've found out to the original image!
			// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			// Create the text we will annotate the box with:
			std::string box_text;
			box_text = cv::format("Prediction = ");
			// Get stringname
			if (prediction >= 0 && prediction <= 1)
			{
				box_text.append(g_listname_t[prediction]);
			}
			else box_text.append("Unknown");
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original,
				box_text,
				cv::Point(pos_x, pos_y),
				cv::FONT_HERSHEY_PLAIN,
				1.0,
				CV_RGB(0, 255, 0), 2.0);
		}

		return original;
	}

private:
	std::vector<cv::Mat> _images;
	std::vector<int> _labels;

	cv::Ptr<cv::face::FaceRecognizer> _model;

	int _im_width;
	int _im_height;

	cv::CascadeClassifier _haar_cascade;

};
