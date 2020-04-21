// put all the necessary includes here, e.g. iostream, fstream, opencv, etc.
// you can use whatever you deem necessary
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <dirent.h>
#include <random>
#include <typeinfo>       // operator typeid

using namespace std;
using namespace cv;

Mat tryLoadImage(const string &filename)
{
	Mat img = imread(filename);
	if (!img.data)
	{
		cout << "ERROR: file " << filename << " not found" << endl;
		cout << "Press enter to exit" << endl;
		cin.get();
		exit(-3);
	}

	// convert to floating point precision
	img.convertTo(img, CV_32FC1);
	return img;
}

void loadLabels(const string &folderPath, vector<Mat> &label_images, vector<string> &label_names, vector<vector<Point2i>> &points_per_class)
{
	struct dirent *entry;
	DIR *dir = opendir(folderPath.c_str());

	if (dir == NULL)
		return;

	std::size_t current = 0;
	int  i=0;
	while ((entry = readdir(dir)) != NULL)
	{
		if (strlen(entry->d_name) < 3) continue; // Ignore current folder (.) and parent folder (..)
		
		label_names.push_back(entry->d_name);	

		Mat image = tryLoadImage(folderPath + label_names[i]);
		label_images.push_back(image);

		vector<Point2i> class_points;
		class_points.clear();
		class_points.reserve(label_images[i].rows * label_images[i].cols);

		for (int row=0; row<label_images[i].rows; row++){
			for (int col=0; col<label_images[i].cols; col++) {
				if (label_images[i].at<float>(row, col) > 0.0f) {
					Point2i new_point(row, col);
					class_points.push_back(new_point);
				}
			}
		}
		points_per_class.push_back(class_points);

		// Delete the .png
		label_names[i] = label_names[i].substr(0, label_names[i].length() - 4); 
		i++;
	}
	closedir(dir);
}


/**
 * Draw N samples from the image.
 * Do not draw samples from a border distance to the edge of te image.
 * Return a vector with the (col, row) positions from the image.
 */
vector<Point2i> draw_samples(vector<Point2i> class_points, int nr_samples, int border) {
	vector<Point2i> samples;
	samples.reserve(nr_samples);
	int nr_samples_drawn = 0;

	if (class_points.size() < nr_samples_drawn) {
		cout << "There are less points than the needed samples.";
		samples.push_back(Point2i(0,0)); // TODO: Segmentation fault here. Find why :) 
		return samples;
	}

	std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, class_points.size()); // define the range

	while (nr_samples_drawn < nr_samples) {
		int i = distr(eng);
		samples.push_back(Point2i(class_points[i].x, class_points[i].y));
		nr_samples_drawn += 1;
	}
	return samples;
}
// feel free to implement your own classes, methods, or whatever you need
// all code has to be submitted, but it can be in multiple files (just make sure it will compile - assuming that the necessary libraries etc. are available)

int main(int argc, char **argv)
{
	CommandLineParser parser(argc, argv, "{@img|img/UH_NAD83-1m.png|input image} {@labels_folder|label/|folder with label images}");
	parser.about("\nThis program demonstrates connected components and use of the trackbar\n");
	parser.printMessage();

	// Read data.
	Mat src = tryLoadImage(samples::findFile(parser.get<String>("@img")));
	cout << "Main image " << src.cols << " x " << src.rows << endl;

	// Read labels.
	int nr_classes = 20; // It can be inputed by the user
	vector<Mat> label_images;
	vector<string> label_names;
	vector<vector<Point2i>> points_per_class;

	label_images.reserve(nr_classes);
	label_names.reserve(nr_classes);
	points_per_class.reserve(nr_classes);

	loadLabels(samples::findFile(parser.get<String>("@labels_folder")), label_images, label_names, points_per_class);

	// Just checking that all works good.
	for (int i=0; i<label_images.size(); i++){
		cout << label_names[i] << " " << label_images[i].cols << " x " << label_images[i].rows << " \t " << points_per_class[i].size() << endl;
	}

	// sample N pixel positions for each class from the image data
	// N can be selected by you (or the user)
	// Note: With C classes, there should be N*C samples
	int nr_samples = 10;
	int M = 20; // size of the patch
	vector<vector<Point2i>> samples;
	samples.reserve(nr_classes);

	for (int i=0; i<label_images.size(); i++) {
		cout << "Draw " << nr_samples << " for class " << label_names[i] << ".\n";
		vector<Point2i> class_samples = draw_samples(points_per_class[i], nr_samples, M);
		samples.push_back(class_samples);
	}

	// extract MxM-sized patches at those pixel positions; M can be selected by you (or the user) but should be larger than 1

	// dimensionality reduction: project image patches into 3D
	// examples: patch re-sizing, PCA (e.g. from opencv), LDA, tSNE (nice tool, available online)

	// visualize 3D data
	// either within the C++ code here or export the 3D data and visualize it outside
	// Does not need to be an interactive 3D visualization. A static 2D projection of the 3D data is sufficient.
}
