#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"

HOG::HOG(std::vector<double> hog)
	: m_HOG(hog) {}

HOG::HOG() {}

namespace HOGParameter
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

	double s_Linear(double magnitude)
	{
		return magnitude;
	}

	double s_Square(double magnitude)
	{
		return magnitude * magnitude;
	}

	double s_SquareRoot(double magnitude)
	{
		return std::sqrt(magnitude);
	}

	std::vector<double> s_L1Norm(std::vector<double> hog)
	{
		double L1Norm = Utilities::s_Epsilon;
		for (int j = 0; j < hog.size(); j++)
			L1Norm += hog[j];

		if (L1Norm == 0)
			throw std::runtime_error("Wut?");

		for (int j = 0; j < hog.size(); j++)
			hog[j] /= L1Norm;

		return hog;
	}

	std::vector<double> s_L2Norm(std::vector<double> hog)
	{
		double L2Norm = Utilities::s_Epsilon;
		for (int j = 0; j < hog.size(); j++)
			L2Norm += hog[j] * hog[j];
		L2Norm = std::sqrt(L2Norm);

		if (L2Norm == 0)
			throw std::runtime_error("Wut?");

		for (int j = 0; j < hog.size(); j++)
			hog[j] /= L2Norm;

		return hog;
	}
}
