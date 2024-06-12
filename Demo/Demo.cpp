#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"
#include "SlidingWindow.h"

int main()
{

	RBFKernelSVM headClassifier = RBFKernelSVM::Load("Model/Head_Classifier");
	RBFKernelSVM handClassifier = RBFKernelSVM::Load("Model/Hand_Classifier");
	RBFKernelSVM torsoClassifier = RBFKernelSVM::Load("Model/Torso_Classifier");
	RBFKernelSVM legClassifier = RBFKernelSVM::Load("Model/Leg_Classifier");
	headClassifier.Log();
	handClassifier.Log();
	torsoClassifier.Log();
	legClassifier.Log();

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



	bool printedReady = false;
	SlidingWindow* sw = nullptr;
	while (true)
	{
		if (Utilities::IsKeyPressed(Utilities::KeyCode::Escape))
			break;

		if (!printedReady)
		{
			printedReady = true;
			std::cout << "Ready!\n";
		}

		if (Utilities::IsKeyPressed(Utilities::KeyCode::Enter))
		{
			std::vector<std::string> paths = Utilities::GetFiles();
			if (paths.empty() || paths[0].empty())
				continue;

			for (auto& path : paths)
			{
				Image image = Image::LoadImage(path);
				image = Utilities::Resize(image, 1600, 1200);

				std::cout << "parsepath = " << Utilities::ParsePath(path) << '\n';

				sw = new SlidingWindow(image, hogParams, 100, 0.9, 2, 0.2, 
					headClassifier, handClassifier, torsoClassifier, legClassifier);

				APPLYTIMER(sw->GenerateBoundingBoxes(), GenerateBoundingBoxes, true);
				//sw->GenerateBoundingBoxes();
				sw->DrawBoundingBox(SlidingWindow::BodyPart::Person, 3);
				//sw->Draw();
				//Utilities::DrawFlush();
				sw->SaveDrawImage("PredictedDrawPerson", Utilities::ParsePath(path));

				sw->ClearDrawnBoundingBox();

				sw->DrawBoundingBox(SlidingWindow::BodyPart::Torso, 3);
				sw->DrawBoundingBox(SlidingWindow::BodyPart::Head, 3);
				sw->DrawBoundingBox(SlidingWindow::BodyPart::Hand, 3);
				sw->DrawBoundingBox(SlidingWindow::BodyPart::Leg, 3);
				//sw->Draw();
				//Utilities::DrawFlush();
				sw->SaveDrawImage("PredictedDrawAll", Utilities::ParsePath(path));


				delete sw;
			}

			printedReady = false;
		}


	}




}