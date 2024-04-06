#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"

HOG::HOG(std::vector<double> hog)
	: m_HOG(hog) {}

HOG::HOG() {}

namespace Kernel
{
	const std::vector<std::vector<double>> s_SobelX =
	{
		{-1, 0, 1},
		{-2, 0, 1},
		{-1, 0, 1},
	};
	const std::vector<std::vector<double>> s_SobelY =
	{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};

	const std::vector<std::vector<double>> s_FDX =
	{
		{0, 0, 0},
		{-1, 0, 1},
		{0, 0, 0},
	};
	const std::vector<std::vector<double>> s_FDY =
	{
		{0, -1, 0},
		{0, 0, 0},
		{0, 1, 0},
	};
}
