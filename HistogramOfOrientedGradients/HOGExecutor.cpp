#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"



void HOGExecutor::ComputeGradients(const std::vector<std::vector<double>>& kernelX, const std::vector<std::vector<double>>& kernelY)
{
	std::vector<double> pixels = m_Image.GetPixels();

	m_CurrentShape = m_Image.GetShape();

	std::vector<double> Dx = Utilities::Convolve(pixels, kernelX, m_CurrentShape);
	std::vector<double> Dy = Utilities::Convolve(pixels, kernelY, m_CurrentShape);

	m_Gradients.clear();
	m_Gradients.reserve(pixels.size() / m_CurrentShape.c);

	for (int i = 0; i < pixels.size(); i += m_CurrentShape.c)
	{
		int maxMagnitudeIndex = -1;
		double maxMagnitudeValue = 0;
		for (int j = 0; j < m_CurrentShape.c; j++)
		{
			double curMagnitude = std::sqrt(Dx[i + j] * Dx[i + j] + Dy[i + j] * Dy[i + j]);
			if (maxMagnitudeValue <= curMagnitude)
			{
				maxMagnitudeValue = curMagnitude;
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

// SingleHOGExecutor ////////////////////////////////////////////////////////////////////////////////////////////////////////////


SingleHOGExecutor::SingleHOGExecutor(Image image)
{
	m_Image = image;
	m_CurrentShape = m_Image.GetShape();
}



// TODO
HOG SingleHOGExecutor::GetHOG()
{
	HOG hog;
	hog.m_HOG.clear();
	hog.m_HOG.resize(m_Blocks.size() * 9 * m_BlockSize * m_BlockSize);

	//::cout << "BLOCKSSIZE = " << m_Blocks.size() << '\n';

	for (int i = 0; i < m_Blocks.size(); i++)
	{
		std::copy(m_Blocks[i].m_HOG.begin(), m_Blocks[i].m_HOG.end(), hog.m_HOG.begin() + i * 9 * m_BlockSize * m_BlockSize);
		//std::cout << "BLOCK HOG SIZE (SHOULD BE 36) = " << m_Blocks[i].m_HOG.size() << '\n';
	}

	//for (int i = 0; i < hog.m_HOG.size(); i++)
	//	std::cout << hog.m_HOG[i] << ", ";

	return hog;
}



void SingleHOGExecutor::Execute()
{
	m_Image = Utilities::Crop(m_Image, s_Margin, s_Margin, m_CurrentShape.x - (2 * s_Margin), m_CurrentShape.y - (2 * s_Margin));

	ComputeGradients(Kernel::s_SobelX, Kernel::s_SobelY);
	m_CellSize = 4;
	m_BlockSize = 2;

	ConstructAndVoteCells();
	ConstructAndNormalizeBlocks();
}




void SingleHOGExecutor::ConstructAndVoteCells()
{
	static constexpr double BinOrientationDifference = 180.0 / 8;
	static constexpr double BinOrientations[] = { 
		0,
		BinOrientationDifference * 1,
		BinOrientationDifference * 2,
		BinOrientationDifference * 3,
		BinOrientationDifference * 4,
		BinOrientationDifference * 5,
		BinOrientationDifference * 6,
		BinOrientationDifference * 7,
		BinOrientationDifference * 8
	};

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
				int index = i + k + j * m_CurrentShape.x;

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
				int index = i + k + j * m_CurrentShape.x;
				//std::cout << "index = " << index << ", flat index = " << 9 * k + 9 * m_BlockSize * j << '\n';
				//std::cout << "m_Cells.size() = " << m_Cells.size() << '\n';
				std::copy(m_Cells[index].m_HOG.begin(), m_Cells[index].m_HOG.end(), HOG.begin() + 9 * k + 9 * m_BlockSize * j);
			}
		}

		// normalize
		double L2Norm = Utilities::s_Epsilon;
		for (int j = 0; j < HOG.size(); j++)
			L2Norm += HOG[j] * HOG[j];
		L2Norm = std::sqrt(L2Norm);

		//for (int l = 0; l < HOG.size(); l++)
		//	std::cout << HOG[l] << ", ";

		if (L2Norm == 0)
			throw std::runtime_error("Wut?");

		for (int j = 0; j < HOG.size(); j++)
			HOG[j] /= L2Norm;

		//for (int l = 0; l < HOG.size(); l++)
		//	std::cout << HOG[l] << ", ";

		//std::cout << "here\n";
		m_Blocks.push_back({ HOG });
		//std::cout << "here2\n";
	}

	std::cout << "BLOCKSSIZE = " << m_Blocks.size() << '\n';
}




Image SingleHOGExecutor::ToImage()
{
	return Image();
}




// SlidingHOGExecutor ////////////////////////////////////////////////////////////////////////////////////////////////////////////


SlidingHOGExecutor::SlidingHOGExecutor(Image image, uint32_t stride)
	: m_Stride(stride)
{
	m_Image = image;
	m_CurrentShape = m_Image.GetShape();

}




void SlidingHOGExecutor::Execute()
{
	//m_Image = Utilities::Crop(m_Image, s_Margin, s_Margin, m_CurrentShape.x - (2 * s_Margin), m_CurrentShape.y - (2 * s_Margin));

	ComputeGradients(Kernel::s_SobelX, Kernel::s_SobelY);
	m_CellSize = 4;
	m_BlockSize = 2;

	ConstructAndVoteCells();
	ConstructAndNormalizeBlocks();

	//std::vector<HOG> hogs = GetHOGs();
	//std::cout << "window count = " << hogs.size() << '\n';
	//std::cout << "hog feature count for each window = " << hogs[0].m_HOG.size();

}




void SlidingHOGExecutor::ConstructAndVoteCells()
{
	static constexpr double BinOrientations[] = { 0, 180.0 * 1 / 8, 180.0 * 2 / 8, 180.0 * 3 / 8, 180.0 * 4 / 8, 180.0 * 5 / 8, 180.0 * 6 / 8, 180.0 * 7 / 8, 180.0 };
	static constexpr double BinOrientationDifference = 180.0 / 8;

	m_Cells.clear();
	m_Cells.reserve((m_CurrentShape.y - m_CellSize + 1) * (m_CurrentShape.x - m_CellSize + 1));

	// recheck the condition of this for loop. 
	for (int i = 0; i < m_Gradients.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);

		//check this if statement
		if (coords.x + m_CellSize > m_CurrentShape.x || coords.y + m_CellSize > m_CurrentShape.y)
			continue;

		std::vector<double> HOG(9, 0.0);
		for (int j = 0; j < m_CellSize; j++)
		{
			for (int k = 0; k < m_CellSize; k++)
			{
				int index = i + k + j * m_CurrentShape.x;

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


// todo
void SlidingHOGExecutor::ConstructAndNormalizeBlocks()
{
	m_Blocks.clear();
	m_Blocks.reserve((m_CurrentShape.y - m_CellSize * m_BlockSize + 1) * (m_CurrentShape.y - m_CellSize * m_BlockSize + 1));

	for (int i = 0; i < m_Gradients.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);

		//check this if statement
		if (coords.x + m_CellSize * m_BlockSize > m_CurrentShape.x || coords.y + m_CellSize * m_BlockSize > m_CurrentShape.y)
			continue;

		std::vector<double> HOG(9 * m_BlockSize * m_BlockSize, 0.0);
		for (int j = 0; j < m_BlockSize; j++)
		{
			for (int k = 0; k < m_BlockSize; k++)
			{
				// check this index. index in gradients vs index in cells
				int index = i - ((i / (m_CurrentShape.c * m_CurrentShape.x)) * (m_CellSize - 1)) + k * m_CellSize + (m_CurrentShape.x - m_CellSize + 1) * j * m_CellSize;// -((i / (m_CurrentShape.c * m_CurrentShape.x)) * (m_CellSize - 1));
				//std::cout << "index = " << index << ", flat index = " << 9 * k + 9 * m_BlockSize * j << '\n';
				//std::cout << "m_Cells.size() = " << m_Cells.size() << '\n';
				std::copy(m_Cells[index].m_HOG.begin(), m_Cells[index].m_HOG.end(), HOG.begin() + 9 * k + 9 * m_BlockSize * j);
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

	//std::cout << "BLOCKSSIZE = " << m_Blocks.size() << '\n';
}

HOG SlidingHOGExecutor::GetHOG(uint32_t flatIndex)
{
	// flatIndex is coords for the detection window
	// index is converted to bloock coords for the detection window inside margin
	Shape coords = Utilities::FlatToCoords(flatIndex, m_CurrentShape);

	//check this if statement
	if (coords.x + s_SlidingWindowSize > m_CurrentShape.x || coords.y + s_SlidingWindowSize > m_CurrentShape.y)
		throw std::runtime_error("SLIDING WINDOW OUTSIDE OF IMAGE");

	// iuhagroiuaeru
	int index = flatIndex - ((flatIndex / (m_CurrentShape.c * m_CurrentShape.x)) * (m_CellSize * m_BlockSize - 1));
	index = index + s_Margin + (m_CurrentShape.x - m_CellSize * m_BlockSize + 1) * s_Margin;

	// hog for current detection window
	HOG hog;
	hog.m_HOG.clear();

	uint64_t rowBlockCount = (((s_SlidingWindowSize - 2 * s_Margin) / m_CellSize) - m_BlockSize + 1);
	hog.m_HOG.resize(rowBlockCount * rowBlockCount * 9 * m_BlockSize * m_BlockSize);


	// COLLECT HOG OF CURRENT DETECTION WINDOW

	for (int j = 0; j < rowBlockCount; j++)
	{
		for (int k = 0; k < rowBlockCount; k++)
		{
			int index2 = index + k * m_CellSize + j * m_CellSize * (m_CurrentShape.x - m_CellSize * m_BlockSize + 1);
			//std::cout << "index = " << index2 << ", flat index = " << (9 * m_BlockSize * m_BlockSize) * (k + rowBlockCount * j) << '\n';
			std::copy(m_Blocks[index2].m_HOG.begin(), m_Blocks[index2].m_HOG.end(), hog.m_HOG.begin() + (9 * m_BlockSize * m_BlockSize) * (k + rowBlockCount * j));
		}
	}

	return hog;
}

HOG SlidingHOGExecutor::GetHOG()
{
	return GetHOG(m_CurrentHOGGetterIndex);
}

std::vector<HOG> SlidingHOGExecutor::GetHOGs()
{
	std::vector<HOG> hogs;
	hogs.reserve((m_CurrentShape.y - s_SlidingWindowSize + 1) * (m_CurrentShape.x - s_SlidingWindowSize + 1));
	// i is coords for the detection window
	// index is coords for the detection window inside margin
	for (int i = 0; i < m_Gradients.size(); i += m_Stride)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);
		//check this if statement
		if (coords.x + s_SlidingWindowSize > m_CurrentShape.x)
		{
			i = m_Stride * m_CurrentShape.x + m_CurrentShape.x * (i / m_CurrentShape.x) - m_Stride;
			continue;
		}
		else if (coords.y + s_SlidingWindowSize > m_CurrentShape.y)
		{
			break;
		}
		std::cout << "i = " << i << '\n';
		hogs.push_back(GetHOG(i));
	}

	return hogs;
}

void SlidingHOGExecutor::AdvanceHOGGetterIndex()
{
	Shape coords = Utilities::FlatToCoords(m_CurrentHOGGetterIndex, m_CurrentShape);
	//check this if statement
	if (coords.x + s_SlidingWindowSize > m_CurrentShape.x)
		m_CurrentHOGGetterIndex = m_Stride * m_CurrentShape.x + m_CurrentShape.x * (m_CurrentHOGGetterIndex / m_CurrentShape.x) - m_Stride;
	else if (coords.y + s_SlidingWindowSize > m_CurrentShape.y)
		m_CurrentHOGGetterIndex = 0;
	else
		m_CurrentHOGGetterIndex += m_Stride;
}

