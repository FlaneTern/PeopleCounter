#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"

int main()
{

	RBFKernelSVM svm = RBFKernelSVM::Load("Model/HeadClassifier");
	svm.Log();

}