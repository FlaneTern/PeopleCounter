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

	std::vector<int> shuffler(features.size());
	std::iota(shuffler.begin(), shuffler.end(), 0);
	//std::shuffle(shuffler.begin(), shuffler.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

	std::vector<std::vector<double>> featuresTemp;
	std::vector<double> labelsTemp;
	for (int i = 0; i < shuffler.size(); i++)
	{
		std::cout << shuffler[i] << '\n';
		featuresTemp.push_back(features[shuffler[i]]);
		labelsTemp.push_back(labels[shuffler[i]]);
	}

	features = featuresTemp;
	labels = labelsTemp;

	std::vector<std::vector<double>> trainFeatures;
	std::vector<double> trainLabels;

	std::vector<std::vector<double>> testFeatures;
	std::vector<double> testLabels;

	for (int i = 0; i < features.size(); i++)
	{
		if ((i % 15) < 12)
		{
			trainFeatures.push_back(features[i]);
			trainLabels.push_back(labels[i]);
		}
		else
		{
			testFeatures.push_back(features[i]);
			testLabels.push_back(labels[i]);
		}
	}

	RBFKernelSVM svm(0.0001, "HeadClassifier");
	svm.Train({ trainFeatures, trainLabels }, 10, 200, 0.00001, 0.1);

	for (int i = 0; i < features.size(); i++)
		std::cout << "Prediction for image " << i << " = " << svm.Predict(features[i]) << "\tTruth = " << labels[i] << '\n';

	svm.Log();

	svm.Save("Model/HeadClassifier");


	return 0;
}