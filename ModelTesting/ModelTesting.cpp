#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"
#include "SlidingWindow.h"

#include "Shader.h"

int main()
{

	std::vector<std::string> bodyParts =
	{
		"Leg",
		"Head",
		"Torso",
		"Hand",
	};

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

	
	RBFKernelSVM headClassifier = RBFKernelSVM::Load("Model/Head_Classifier");
	RBFKernelSVM handClassifier = RBFKernelSVM::Load("Model/Hand_Classifier");
	RBFKernelSVM torsoClassifier = RBFKernelSVM::Load("Model/Torso_Classifier");
	RBFKernelSVM legClassifier = RBFKernelSVM::Load("Model/Leg_Classifier");
	
	for (auto& bodyPart : bodyParts)
	{
		std::vector<std::string> positiveImagesPath = Utilities::GetFilesInDirectory("Dataset_Final/Testing/" + bodyPart + "/Positive");
		std::vector<Image> positiveImages = Image::LoadImages(positiveImagesPath);

		std::vector<std::string> negativeImagesPath = Utilities::GetFilesInDirectory("Dataset_Final/Testing/" + bodyPart + "/Negative");
		std::vector<Image> negativeImages = Image::LoadImages(negativeImagesPath);


		std::vector<std::vector<double>> features;
		features.reserve(positiveImages.size() + negativeImages.size());

		std::vector<double> labels;
		labels.reserve(positiveImages.size() + negativeImages.size());

		std::vector<double> labelsPrediction;
		labels.reserve(positiveImages.size() + negativeImages.size());




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

		for (int i = 0; i < features.size(); i++)
		{
			std::cout << "Predicting Image " << i + 1 << " out of " << features.size() << '\n';
		
			if (bodyPart == "Torso")
				labelsPrediction.push_back(torsoClassifier.Predict(features[i]) >= 0 ? 1 : -1);
			else if (bodyPart == "Head")
				labelsPrediction.push_back(headClassifier.Predict(features[i]) >= 0 ? 1 : -1);
			else if (bodyPart == "Hand")
				labelsPrediction.push_back(handClassifier.Predict(features[i]) >= 0 ? 1 : -1);
			else if (bodyPart == "Leg")
				labelsPrediction.push_back(legClassifier.Predict(features[i]) >= 0 ? 1 : -1);

		}

		int tp = 0;
		int fp = 0;
		int fn = 0;
		int tn = 0;

		for (int i = 0; i < features.size(); i++)
		{
			if (labels[i] == 1 && labelsPrediction[i] == 1)
				tp++;
			else if (labels[i] == -1 && labelsPrediction[i] == 1)
				fp++;
			else if (labels[i] == 1 && labelsPrediction[i] == -1)
				fn++;
			else
				tn++;
		}

		std::cout << "TP = " << tp << '\n';
		std::cout << "FP = " << fp << '\n';
		std::cout << "FN = " << fn << '\n';
		std::cout << "TN = " << tn << '\n';

		std::cout << "Accuracy = " << ((double)tp + tn) / (tp + tn + fp + fn) << '\n';
		std::cout << "Precision = " << (double)tp / (tp + fp) << '\n';
		std::cout << "Recall = " << (double)tp / (tp + fn) << '\n';
		std::cout << "F1 Score = " << 2 * (double)tp / (2 * tp + fp + fn) << '\n';

		std::cout << '\n';
	}

	std::vector<std::string> classroomImagesPath = Utilities::GetFilesInDirectory("Dataset_Final/Testing/Classroom/");

	for (auto& classroomImagePath : classroomImagesPath)
	{
		Image image = Image::LoadImage(classroomImagePath);
		image = Utilities::Resize(image, 1600, 1200);
		SlidingWindow sw(image, hogParams, 100, 0.9, 2, 0.2, 
			headClassifier, handClassifier, torsoClassifier, legClassifier);

		APPLYTIMER(sw.GenerateBoundingBoxes(), GenerateBoundingBoxes, true);
		
		sw.DrawBoundingBox(SlidingWindow::BodyPart::Person, 3);
		sw.SaveDrawImage("PredictedDrawPerson");

		sw.ClearDrawnBoundingBox();

		sw.DrawBoundingBox(SlidingWindow::BodyPart::Torso, 3);
		sw.DrawBoundingBox(SlidingWindow::BodyPart::Head, 3);
		sw.DrawBoundingBox(SlidingWindow::BodyPart::Hand, 3);
		sw.DrawBoundingBox(SlidingWindow::BodyPart::Leg, 3);
		sw.SaveDrawImage("PredictedDrawAll");

		//sw.Draw();
		//Utilities::DrawFlush();

	}
	

}