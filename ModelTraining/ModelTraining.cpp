#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"

int main()
{

	std::vector<std::string> bodyParts =
	{
		"Head",
		"Torso",
		"Hand",
		"Leg",
	};

	for (auto& bodyPart : bodyParts)
	{
		std::vector<std::string> positiveImagesPath = Utilities::GetFilesInDirectory("Dataset_Final/Training/" + bodyPart + "/Positive");
		std::vector<Image> positiveImages = Image::LoadImages(positiveImagesPath);

		std::vector<std::string> negativeImagesPath = Utilities::GetFilesInDirectory("Dataset_Final/Training/" + bodyPart + "/Negative");
		std::vector<Image> negativeImages = Image::LoadImages(negativeImagesPath);

		std::vector<std::vector<double>> features;
		features.reserve(positiveImages.size() + negativeImages.size());

		std::vector<double> labels;
		labels.reserve(positiveImages.size() + negativeImages.size());

		HOGParameter::Parameters hogParams =
		{
			128,
			16,
			HOGParameter::s_SobelX,
			HOGParameter::s_SobelY,
			4,
			2,
			false,
			9,
			HOGParameter::s_Linear, HOGParameter::s_L2Norm
		};


		for (int i = 0; i < positiveImages.size(); i++)
		{
			SingleHOGExecutor hoge(positiveImages[i], hogParams);
			hoge.Execute();
			features.push_back(hoge.GetHOG().m_HOG);
			labels.push_back(1);
		}

		for (int i = 0; i < negativeImages.size(); i++)
		{
			SingleHOGExecutor hoge(negativeImages[i], hogParams);
			hoge.Execute();
			features.push_back(hoge.GetHOG().m_HOG);
			labels.push_back(-1);
		}

		// shuffling
		{
			std::vector<int> shuffler(features.size());
			std::iota(shuffler.begin(), shuffler.end(), 0);
			std::shuffle(shuffler.begin(), shuffler.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

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
		}


		RBFKernelSVM svm(0.0001, bodyPart + "_Classifier");
		svm.Train({ features, labels }, 10, 200, 0.00001, 0.1);

		svm.Log();

		svm.Save("Model/" + svm.GetName());
	}



		
	return 0;
}