#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"
#include "SlidingWindow.h"

int main()
{

	RBFKernelSVM svm = RBFKernelSVM::Load("Model/HeadClassifier");
	svm.Log();

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

	bool imageLoaded = false;
	SlidingWindow* sw = nullptr;
	while (!Utilities::IsKeyPressed(Utilities::KeyCode::Escape))
	{

		if (Utilities::IsKeyPressed(Utilities::KeyCode::Enter))
		{
			std::string path = Utilities::GetFile();
			if (path.empty())
				continue;

			Image image = Image::LoadImage(path);
			image = Utilities::Resize(image, 128, 128);

			sw = new SlidingWindow(image, hogParams, 1, 0.9, 3, 0.5, svm, svm, svm, svm);

			APPLYTIMER(sw->GenerateBoundingBoxes(), GenerateBoundingBoxes, true);
			//sw->GenerateBoundingBoxes();
			sw->DrawBoundingBox(SlidingWindow::BodyPart::Person, 3);
			sw->DrawBoundingBox(SlidingWindow::BodyPart::Torso, 3);
			imageLoaded = true;
		}

		if (imageLoaded)
		{
			sw->Draw();
			Utilities::DrawFlush();
			imageLoaded = false;
			delete sw;
		}
	}




}