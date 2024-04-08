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

	Image image = Image::LoadImage("Testing/Classroom/IMG_2473Edited.png");
	image = Utilities::Resize(image, 150, 150);

	SlidingWindow sw(image, hogParams, 3, 0.9, 3, 0.5, svm, svm, svm, svm);
	sw.GenerateBoundingBoxes();
	sw.DrawBoundingBox(SlidingWindow::BodyPart::Hand, 3);
	sw.Draw();

	Utilities::DrawFlush();

	

}