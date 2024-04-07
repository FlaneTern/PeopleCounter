#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"

int main()
{
	std::vector<std::string> handImagesPath = Utilities::GetFilesInDirectory("Testing/Hand/");
	std::vector<Image> handImages = Image::LoadImages(handImagesPath);

	std::vector<std::string> headImagesPath = Utilities::GetFilesInDirectory("Testing/Head/");
	std::vector<Image> headImages = Image::LoadImages(headImagesPath);

	std::vector<std::vector<double>> features;
	features.reserve(handImages.size() + headImages.size());

	std::vector<double> labels;
	labels.reserve(handImages.size() + headImages.size());

	for (int i = 0; i < handImages.size(); i++)
	{
		handImages[i] = Utilities::Resize(handImages[i], 128, 128);
		//handImages[i] = Utilities::GaussianBlur(handImages[i], 0.8, 5);
		SingleHOGExecutor hoge(handImages[i]);
		hoge.Execute();
		features.push_back(hoge.GetHOG().m_HOG);
		labels.push_back(1);
	}

	for (int i = 0; i < headImages.size(); i++)
	{
		headImages[i] = Utilities::Resize(headImages[i], 128, 128);
		//headImages[i] = Utilities::GaussianBlur(headImages[i], 0.8, 5);
		SingleHOGExecutor hoge(headImages[i]);
		hoge.Execute();
		features.push_back(hoge.GetHOG().m_HOG);
		labels.push_back(-1);
	}

	RBFKernelSVM svm(1.0, "HeadClassifier");
	svm.Train({ features, labels }, 1, 100, 0.001, 0.01);

	for (int i = 0; i < features.size(); i++)
		std::cout << "Prediction for image " << i << " = " << svm.Predict(features[i]) << "\tTruth = " << labels[i] << '\n';

	svm.Log();

	return 0;
}