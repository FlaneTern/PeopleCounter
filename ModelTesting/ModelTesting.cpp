#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"
#include "SlidingWindow.h"

#include "Shader.h"

int main()
{

#if 0
	std::vector<std::vector<double>> modelTorsoData = 
	{
		{ 1.0, 2.0 },
		{ 2.0, 3.0 },
	};
	std::vector<std::vector<double>> modelHeadData = 
	{
		{ 1.0, 2.0 },
		{ 2.0, 3.0 },
		{ 2.0, 3.0 },
		{ 2.0, 3.0 },
	};
	std::vector<std::vector<double>> modelHandData = 
	{
		{ 1.0, 2.0 },
		{ 2.0, 3.0 },
	};
	std::vector<std::vector<double>> modelLegData = 
	{
		{ 1.0, 2.0 },
		{ 1.0, 2.0 },
		{ 1.0, 2.0 },
		{ 2.0, 3.0 },
	};

	modelHeadData = {};
	for (int i = 0; i < 780; i++)
		modelHeadData.push_back({ 2.0, 3.0 });

	std::vector<double> inputTorsoData = { 1.0, 2.0 };
	std::vector<double> inputHeadData = { 3.0, 4.0 };
	std::vector<double> inputHandData = { 5.0, 6.0 };
	std::vector<double> inputLegData = { 7.0, 8.0 };

	Shader::Initialize(modelTorsoData, modelHeadData, modelHandData, modelLegData);
	


	Shader::DispatchComputeShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Torso], Shader::SSBOIndex::InputTorso, inputTorsoData, modelTorsoData.size());
	Shader::DispatchComputeShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Head], Shader::SSBOIndex::InputHead, inputHeadData, modelHeadData.size());
	Shader::DispatchComputeShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Hand], Shader::SSBOIndex::InputHand, inputHandData, modelHandData.size());
	Shader::DispatchComputeShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Leg], Shader::SSBOIndex::InputLeg, inputHandData, modelLegData.size());

	
	return 0;

#endif


	std::vector<std::string> bodyParts =
	{
		"Leg",
		"Head",
		"Torso",
		"Hand",
	};

	HOGParameter::Parameters hogParams =
	{
		//128,
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
	
#if 11
	//auto temp1 = torsoClassifier.GetDataFeatures();
	//auto temp2 = headClassifier.GetDataFeatures();
	//auto temp3 = handClassifier.GetDataFeatures();
	//auto temp4 = legClassifier.GetDataFeatures();
	//Shader::Initialize(temp1, temp2, temp3, temp4);
	//
	//torsoClassifier.SetShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Torso], (unsigned int)Shader::SSBOIndex::InputTorso);
	//headClassifier.SetShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Head], (unsigned int)Shader::SSBOIndex::InputHead);
	//handClassifier.SetShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Hand], (unsigned int)Shader::SSBOIndex::InputHand);
	//legClassifier.SetShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Leg], (unsigned int)Shader::SSBOIndex::InputLeg);

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
			//if(bodyPart == "Torso")
			//	labelsPrediction.push_back(torsoClassifier.PredictShader(features[i], Shader::CSP[(unsigned int)Shader::CSPIndex::Torso], (unsigned int)Shader::SSBOIndex::InputTorso) >= 0 ? 1 : -1);
			//else if(bodyPart == "Head")
			//	labelsPrediction.push_back(headClassifier.PredictShader(features[i], Shader::CSP[(unsigned int)Shader::CSPIndex::Head], (unsigned int)Shader::SSBOIndex::InputHead) >= 0 ? 1 : -1);
			//else if (bodyPart == "Hand")
			//	labelsPrediction.push_back(handClassifier.PredictShader(features[i], Shader::CSP[(unsigned int)Shader::CSPIndex::Hand], (unsigned int)Shader::SSBOIndex::InputHand) >= 0 ? 1 : -1);
			//else if (bodyPart == "Leg")
			//	labelsPrediction.push_back(legClassifier.PredictShader(features[i], Shader::CSP[(unsigned int)Shader::CSPIndex::Leg], (unsigned int)Shader::SSBOIndex::InputLeg) >= 0 ? 1 : -1);
		
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

		//auto temp = Shader::DispatchComputeShader(Shader::CSP[(unsigned int)Shader::CSPIndex::Leg], Shader::SSBOIndex::InputLeg, features[6], legClassifier.m_LagrangeMultipliers.size());
		//std::cout << "LAGRANGEMULTIPLIERSIZE" << temp[0] << '\n';
		//std::cout << "FEATURECOUNT" << temp[1] << '\n';
		//for(int i = 2; i < temp.size(); i++)
		//	std::cout << temp[i] << ',';
		std::cout << '\n';
		//Image image = Image::LoadImage("Testing/Classroom/IMG_2473Edited.png");
		//image = Utilities::Resize(image, 250, 250);
		//
		//SlidingWindow sw(image, hogParams, 5, 0.9, 3, 0.5, svm, svm, svm, svm);
		//sw.GenerateBoundingBoxes();
		//sw.DrawBoundingBox(SlidingWindow::BodyPart::Person, 3);
		//sw.Draw();
		//
		//Utilities::DrawFlush();
	}
#endif

#if 1
	std::vector<std::string> classroomImagesPath = Utilities::GetFilesInDirectory("Dataset_Final/Testing/Classroom/");

	for (auto& classroomImagePath : classroomImagesPath)
	{
		Image image = Image::LoadImage(classroomImagePath);
		image = Utilities::Resize(image, 1600, 1200);
		SlidingWindow sw(image, hogParams, 100, 0.9, 2, 0.2, 
			headClassifier, handClassifier, torsoClassifier, legClassifier);

		APPLYTIMER(sw.GenerateBoundingBoxes(), GenerateBoundingBoxes, true);
		
		//sw->GenerateBoundingBoxes();
		sw.DrawBoundingBox(SlidingWindow::BodyPart::Person, 3);
		//sw.DrawBoundingBox(SlidingWindow::BodyPart::Torso, 3);
		//sw.DrawBoundingBox(SlidingWindow::BodyPart::Head, 3);
		//sw.DrawBoundingBox(SlidingWindow::BodyPart::Hand, 3);
		//sw.DrawBoundingBox(SlidingWindow::BodyPart::Leg, 3);

		//sw.Draw();
		//sw.SaveDrawImage();
		//Utilities::DrawFlush();

	}
#endif
	

}