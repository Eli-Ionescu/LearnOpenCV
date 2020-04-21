// put all the necessary includes here, e.g. iostream, fstream, opencv, etc.
// you can use whatever you deem necessary
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <dirent.h>
#include <random>
#include <typeinfo> // operator typeid

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
	int i = 0;
	while ((entry = readdir(dir)) != NULL)
	{
		if (strlen(entry->d_name) < 3)
			continue; // Ignore current folder (.) and parent folder (..)

		label_names.push_back(entry->d_name);

		Mat image = tryLoadImage(folderPath + label_names[i]);
		label_images.push_back(image);

		vector<Point2i> class_points;
		class_points.clear();
		class_points.reserve(label_images[i].rows * label_images[i].cols);

		for (int row = 0; row < label_images[i].rows; row++)
		{
			for (int col = 0; col < label_images[i].cols; col++)
			{
				if (label_images[i].at<float>(row, col) > 0.0f)
				{
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
vector<Point2i> draw_samples(vector<Point2i> class_points, int nr_samples, int border, int rows, int cols)
{
	vector<Point2i> samples;
	samples.reserve(nr_samples);
	int nr_samples_drawn = 0;

	std::random_device rd;										   // obtain a random number from hardware
	std::mt19937 eng(rd());										   // seed the generator
	std::uniform_int_distribution<> distr(0, class_points.size()); // define the range

	while (nr_samples_drawn < nr_samples)
	{
		int i = distr(eng);
		Point2i new_sample(class_points[i].x, class_points[i].y);

		// Make sure that the samples point is not on a border.
		// This is needed in order to extract the patches.
		if (new_sample.x < border || new_sample.y < border ||
			new_sample.x > rows - border || new_sample.y > cols - border)
			continue;
		samples.push_back(new_sample);
		nr_samples_drawn += 1;
	}
	return samples;
}

/**
 * Extract patches of size M ardound each point.
 */
vector<Mat> get_patches(Mat image, vector<Point2i> points, int M)
{
	vector<Mat> patches;
	patches.reserve(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		Mat_<Vec3f> new_patch(M, M);

		int start_r = points[i].x - M / 2;
		int end_r = points[i].x + M / 2;

		int start_c = points[i].y - M / 2;
		int end_c = points[i].y + M / 2;


		if (start_c < 0 || start_r < 0 ||
			end_r > image.rows || end_c > image.cols)
		{
			cout << "The point is too close to the ege of the image." << endl;
			cout << "Cannot extract patch for " << points[i].x << " x " << points[i].y << endl;
			continue;
		}

		int row = 0;
		int col = 0;
		for (int r = start_r; r < end_r; r++)
		{
			col = 0;
			for (int c = start_c; c < end_c; c++)
			{
				new_patch.at<Vec3f>(row, col) = image.at<Vec3f>(r, c);
				col++;
			}
			row++;
		}
		patches.push_back(new_patch);
	}
	return patches;
}


/*****************  Functions taken from the internet ************************/
/* 				https://bytefish.de/blog/pca_in_opencv/                  	 */


// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            // CV_Error(CV_StsBadArg, error_message);
			cout << error_message;
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            // CV_Error(CV_StsBadArg, error_message);
			cout << error_message;
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

/*********************************************************************************/

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
	for (int i = 0; i < label_images.size(); i++)
	{
		cout << label_names[i] << " " << label_images[i].cols << " x " << label_images[i].rows << " \t " << points_per_class[i].size() << endl;
	}

	// sample N pixel positions for each class from the image data
	// N can be selected by you (or the user)
	// Note: With C classes, there should be N*C samples
	int nr_samples = 10;
	int M = 10; // size of the patch
	vector<vector<Point2i>> samples;
	samples.reserve(nr_classes);

	for (int i = 0; i < label_images.size(); i++)
	{
		if (points_per_class[i].size() < nr_samples)
		{
			cout << label_names[i] << " has only " << points_per_class[i].size() << " samples. Cannot draw samples :( " << endl;
			vector<Point2i> no_samples;
			samples.push_back(no_samples);
			continue;
		}
		cout << "Draw " << nr_samples << "points for class " << label_names[i] << ".\n";
		vector<Point2i> class_samples = draw_samples(points_per_class[i], nr_samples, M, label_images[i].rows, label_images[i].cols);
		samples.push_back(class_samples);
	}

	// extract MxM-sized patches at those pixel positions; M can be selected by you (or the user) but should be larger than 1
	vector<vector<Mat>> patches;
	patches.reserve(nr_classes);

	for (int i = 0; i < points_per_class.size(); i++)
	{
		cout << "Getting patches for " << label_names[i] << "...    ";
		vector<Mat> class_patches;
		class_patches.reserve(nr_samples);
		class_patches = get_patches(src, samples[i], M);
		cout << class_patches.size() << " patches extracted." << endl;
 		patches.push_back(class_patches);
	}

	cout << patches.size() << endl;

	// dimensionality reduction: project image patches into 3D
	// examples: patch re-sizing, PCA (e.g. from opencv), LDA, tSNE (nice tool, available online)
	// For now, the data is stored like this:
	// nr_clases(C) vectors, that contain nr_samples(N) Mat, each Mat of size MxM
	// In order to use PCA, there is needed 
	vector<Mat> all_patches;
	all_patches.reserve(nr_classes * nr_samples);

	vector<string> patch_labels;
	patches.reserve(nr_classes * nr_samples);
	for (int i = 0; i < patches.size(); i++) {
		for (int j = 0; j < patches[i].size(); j++) {
			all_patches.push_back(patches[i][j]);
			patch_labels.push_back(label_names[i]);
		}
	}
	cout << all_patches.size() << endl;

	Mat data = asRowMatrix(all_patches, CV_32FC1);

	 // Number of components to keep for the PCA:
    int num_components = 3;

    // Perform a PCA:
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, num_components); // TODO(check how the data is handled row / col);

    // And copy the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();

	cout << "Data: " << data.rows << " x " << data.cols << endl;
	cout << "Mean: " << mean.rows << " x " << mean.cols << endl;
	cout << "eigenvals " << eigenvalues.rows << " x " << eigenvalues.cols << endl;
	cout << "eigenvectors " << eigenvectors.rows << " x " << eigenvectors.cols << endl;

	for (int c=0; c< data.cols; c++) {
		cout << data.at<Vec3f>(0, c) << endl;
	}

    // // The mean face:
    // imshow("avg", norm_0_255(mean.reshape(1, all_patches[0].rows)));

    // // The first three eigenfaces:
    // imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, all_patches[0].rows));
    // imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, all_patches[0].rows));
    // imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, all_patches[0].rows));

    // // // Show the images:
    // waitKey(0);

}
