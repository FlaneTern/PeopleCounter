#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"

SingleHOGExecutor::SingleHOGExecutor(Image image)
{
	m_Image = image;
	m_CurrentShape = m_Image.GetShape();
}

HOG HOGExecutor::Execute()
{
	ComputeGradients(Kernel::s_SobelX, Kernel::s_SobelY);
	m_CellSize = 4;
	m_BlockSize = 2;

	ConstructAndVoteCells();
	ConstructAndNormalizeBlocks();

	return GetHOG();
}

void HOGExecutor::ComputeGradients(const std::vector<std::vector<double>>& kernelX, const std::vector<std::vector<double>>& kernelY)
{
	std::vector<double> pixels = m_Image.GetPixels();

	m_CurrentShape = m_Image.GetShape();

	std::vector<double> Dx = Utilities::Convolve(pixels, kernelX, m_CurrentShape);
	std::vector<double> Dy = Utilities::Convolve(pixels, kernelY, m_CurrentShape);

	m_Gradients.clear();
	m_Gradients.reserve(pixels.size() / m_CurrentShape.c);

	std::vector<double> curGradientsMagnitudes;
	for (int i = 0; i < pixels.size(); i += m_CurrentShape.c)
	{
		curGradientsMagnitudes.clear();
		curGradientsMagnitudes.reserve(m_CurrentShape.c);

		for (int channel = 0; channel < m_CurrentShape.c; channel++)
			curGradientsMagnitudes.push_back(std::sqrt(Dx[i + channel] * Dx[i + channel] + Dy[i + channel] * Dy[i + channel]));

		int maxMagnitudeIndex = -1;
		double maxMagnitudeValue = 0;
		for (int j = 0; j < m_CurrentShape.c; j++)
		{
			if (maxMagnitudeValue <= curGradientsMagnitudes[j])
			{
				maxMagnitudeValue = curGradientsMagnitudes[j];
				maxMagnitudeIndex = j;
			}
		}
		double orientation = (double)(std::atan2(Dy[i + maxMagnitudeIndex], Dx[i + maxMagnitudeIndex]) * 180.0 / std::numbers::pi);
		if (orientation < 0)
			orientation += 180;
		m_Gradients.push_back({ maxMagnitudeValue, orientation });

	}

	m_CurrentShape.c = 1;
}

// TODO
HOG HOGExecutor::GetHOG()
{
	HOG hog;
	hog.m_HOG.clear();
	hog.m_HOG.resize(m_Blocks.size() * 9 * m_BlockSize * m_BlockSize);

	for (int i = 0; i < m_Blocks.size(); i++)
		std::copy(m_Blocks[i].m_HOG.begin(), m_Blocks[i].m_HOG.end(), hog.m_HOG.begin() + i * 9 * m_BlockSize);

	return hog;
}

void SlidingHOGExecutor::ConstructAndVoteCells()
{
	static constexpr double BinOrientations[] = { 0, 180.0 * 1 / 8, 180.0 * 2 / 8, 180.0 * 3 / 8, 180.0 * 4 / 8, 180.0 * 5 / 8, 180.0 * 6 / 8, 180.0 * 7 / 8, 180.0 };
	static constexpr double BinOrientationDifference = 180.0 / 8;

	m_Cells.clear();
	m_Cells.reserve(m_CurrentShape.y * m_CurrentShape.x);

	for (int i = 0; i < m_Gradients.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);
		if (coords.x + m_CellSize >= m_CurrentShape.x || coords.y + m_CellSize >= m_CurrentShape.y)
		{
			m_Cells.push_back({ {} });
			continue;
		}

		std::vector<double> HOG(9, 0.0);
		for (int j = 0; j < m_CellSize; j++)
		{
			for (int k = 0; k < m_CellSize; k++)
			{
				int index = i + j + m_CurrentShape.x * k;
				int lowerBin = (int)(m_Gradients[index].Orientation / BinOrientationDifference);

				double voteMagnitude = m_Gradients[index].Magnitude; // change this for other functions
				double upperBinVote = voteMagnitude * (m_Gradients[index].Orientation - BinOrientations[lowerBin]) / (BinOrientationDifference);

				HOG[lowerBin] += voteMagnitude - upperBinVote;
				if (lowerBin < 8)
					HOG[lowerBin + 1] += upperBinVote;
			}
		}

		m_Cells.push_back({ HOG });

	}
}

void SlidingHOGExecutor::ConstructAndNormalizeBlocks()
{
	for (int i = 0; i < m_Gradients.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);
		if (coords.x + m_CellSize * m_BlockSize >= m_CurrentShape.x || coords.y + m_CellSize * m_BlockSize >= m_CurrentShape.y)
		{
			m_Blocks.push_back({ {} });
			continue;
		}

		std::vector<double> HOG(9 * m_BlockSize * m_BlockSize, 0.0);
		for (int j = 0; j < m_BlockSize; j++)
		{
			for (int k = 0; k < m_BlockSize; k++)
			{
				int index = i + j * m_CellSize + m_CurrentShape.x * k * m_CellSize;
				std::copy(m_Cells[index].m_HOG.begin(), m_Cells[index].m_HOG.end(), HOG.begin() + 9 * j + 9 * m_BlockSize * k);
			}
		}

		// normalize
		double L2Norm = 0.0000000001;
		for (int j = 0; j < HOG.size(); j++)
			L2Norm += HOG[j] * HOG[j];
		L2Norm = std::sqrt(L2Norm);

		for (int j = 0; j < HOG.size(); j++)
			HOG[j] /= L2Norm;

		m_Blocks.push_back({ HOG });
	}
}

HOG SingleHOGExecutor::Execute()
{


	m_Image = Utilities::Crop(m_Image, s_Margin, s_Margin, m_CurrentShape.x - (2 * s_Margin), m_CurrentShape.y - (2 * s_Margin));

	ComputeGradients(Kernel::s_SobelX, Kernel::s_SobelY);
	m_CellSize = 4;
	m_BlockSize = 2;

	ConstructAndVoteCells();
	ConstructAndNormalizeBlocks();

	return GetHOG();
}


void SingleHOGExecutor::ConstructAndVoteCells()
{
	static constexpr double BinOrientations[] = { 0, 180.0 * 1 / 8, 180.0 * 2 / 8, 180.0 * 3 / 8, 180.0 * 4 / 8, 180.0 * 5 / 8, 180.0 * 6 / 8, 180.0 * 7 / 8, 180.0 };
	static constexpr double BinOrientationDifference = 180.0 / 8;

	m_Cells.clear();
	m_Cells.reserve(m_CurrentShape.y * m_CurrentShape.x / (m_CellSize * m_CellSize));

	// recheck the condition of this for loop. 
	for (int i = 0; i < m_Gradients.size() - ((m_CellSize - 1) * m_CurrentShape.x); i += m_CellSize)
	{

		std::vector<double> HOG(9, 0.0);
		for (int j = 0; j < m_CellSize; j++)
		{
			for (int k = 0; k < m_CellSize; k++)
			{
				int index = i + j * m_CurrentShape.x + k;

				int lowerBin = (int)(m_Gradients[index].Orientation / BinOrientationDifference);

				double voteMagnitude = m_Gradients[index].Magnitude; // change this for other functions
				double upperBinVote = voteMagnitude * (m_Gradients[index].Orientation - BinOrientations[lowerBin]) / (BinOrientationDifference);

				HOG[lowerBin] += voteMagnitude - upperBinVote;
				if (lowerBin < 8)
					HOG[lowerBin + 1] += upperBinVote;
			}
		}

		m_Cells.push_back({ HOG });

		if ((i + m_CellSize) % m_CurrentShape.x == 0)
			i += (m_CellSize - 1) * m_CurrentShape.x;

	}

	m_CurrentShape.x /= m_CellSize;
	m_CurrentShape.y /= m_CellSize;
	m_CurrentShape.c = 1;
}

void SingleHOGExecutor::ConstructAndNormalizeBlocks()
{
	m_Blocks.clear();
	m_Blocks.reserve((m_CurrentShape.y - m_BlockSize + 1) * (m_CurrentShape.x - m_BlockSize + 1));
	for (int i = 0; i < m_Cells.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);
		if (coords.x + m_BlockSize - 1 >= m_CurrentShape.x || coords.y + m_BlockSize - 1 >= m_CurrentShape.y)
			continue;

		std::vector<double> HOG(9 * m_BlockSize * m_BlockSize, 0.0);
		for (int j = 0; j < m_BlockSize; j++)
		{
			for (int k = 0; k < m_BlockSize; k++)
			{
				int index = i + j + m_CurrentShape.x * k;
				//std::cout << "index = " << index << ", flat index = " << 9 * j + 9 * m_BlockSize * k << '\n';
				//std::cout << "m_Cells.size() = " << m_Cells.size() << '\n';
				std::copy(m_Cells[index].m_HOG.begin(), m_Cells[index].m_HOG.end(), HOG.begin() + 9 * j + 9 * m_BlockSize * k);
			}
		}

		// normalize
		double L2Norm = Utilities::s_Epsilon;
		for (int j = 0; j < HOG.size(); j++)
			L2Norm += HOG[j] * HOG[j];
		L2Norm = std::sqrt(L2Norm);

		for (int j = 0; j < HOG.size(); j++)
			HOG[j] /= L2Norm;

		//std::cout << "here\n";
		m_Blocks.push_back({ HOG });
		//std::cout << "here2\n";
	}

	std::cout << "BLOCKSSIZE = " << m_Blocks.size() << '\n';
}

HOG SingleHOGExecutor::GetHOG()
{
	HOG hog;
	hog.m_HOG.clear();
	hog.m_HOG.resize(m_Blocks.size() * 9 * m_BlockSize * m_BlockSize);

	for (int i = 0; i < m_Blocks.size(); i++)
		std::copy(m_Blocks[i].m_HOG.begin(), m_Blocks[i].m_HOG.end(), hog.m_HOG.begin() + i * 9 * m_BlockSize);

	return hog;
}

Image SingleHOGExecutor::ToImage()
{
	return Image();
}