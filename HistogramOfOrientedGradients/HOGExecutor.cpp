#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"


static std::vector<double> s_CreateOrientationBins(bool signedOrientationBins, uint32_t binCount)
{
	double deg = signedOrientationBins ? 360 : 180;

	std::vector<double> binOrientations;
	binOrientations.reserve(binCount);

	for (int i = 0; i < binCount; i++)
		binOrientations.push_back(i * deg / (double)(binCount - 1));

	return binOrientations;
}

void HOGExecutor::ComputeGradients()
{
	std::vector<double> pixels = m_Image.GetPixels();

	m_CurrentShape = m_Image.GetShape();

	std::vector<double> Dx = Utilities::Convolve(pixels, m_Params.XKernel, m_CurrentShape);
	std::vector<double> Dy = Utilities::Convolve(pixels, m_Params.YKernel, m_CurrentShape);

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
		{
			if (!m_Params.SignedOrientationBins)
				orientation += 180;
			else
				orientation += 360;
		}
		m_Gradients.push_back({ maxMagnitudeValue, orientation });

	}

	m_CurrentShape.c = 1;
}

// SingleHOGExecutor ////////////////////////////////////////////////////////////////////////////////////////////////////////////


SingleHOGExecutor::SingleHOGExecutor(Image image, HOGParameter::Parameters params)
{
	m_Params = params;
	m_Image = image;
	m_CurrentShape = m_Image.GetShape();
}



// TODO
HOG SingleHOGExecutor::GetHOG()
{
	HOG hog;
	hog.m_HOG.clear();
	hog.m_HOG.resize(m_Blocks.size() * m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize);

	//::cout << "BLOCKSSIZE = " << m_Blocks.size() << '\n';

	for (int i = 0; i < m_Blocks.size(); i++)
	{
		std::copy(m_Blocks[i].m_HOG.begin(), m_Blocks[i].m_HOG.end(), hog.m_HOG.begin() + i * m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize);
		//std::cout << "BLOCK HOG SIZE (SHOULD BE 36) = " << m_Blocks[i].m_HOG.size() << '\n';
	}

	//for (int i = 0; i < hog.m_HOG.size(); i++)
	//	std::cout << hog.m_HOG[i] << ", ";

	return hog;
}



void SingleHOGExecutor::Execute()
{

	m_Image = Utilities::Crop(m_Image, m_Params.Margin, m_Params.Margin, m_CurrentShape.x - (2 * m_Params.Margin), m_CurrentShape.y - (2 * m_Params.Margin));

	ComputeGradients();

	ConstructAndVoteCells();
	ConstructAndNormalizeBlocks();
}




void SingleHOGExecutor::ConstructAndVoteCells()
{
	// create the bins
	std::vector<double> binOrientation = s_CreateOrientationBins(m_Params.SignedOrientationBins, m_Params.BinCount);
	double binOrientationDifference = (m_Params.SignedOrientationBins ? 360 : 180) / (double)(m_Params.BinCount - 1);

	m_Cells.clear();
	m_Cells.reserve(m_CurrentShape.y * m_CurrentShape.x / (m_Params.CellSize * m_Params.CellSize));

	// recheck the condition of this for loop. 
	for (int i = 0; i < m_Gradients.size() - ((m_Params.CellSize - 1) * m_CurrentShape.x); i += m_Params.CellSize)
	{

		std::vector<double> HOG(m_Params.BinCount, 0.0);
		for (int j = 0; j < m_Params.CellSize; j++)
		{
			for (int k = 0; k < m_Params.CellSize; k++)
			{
				int index = i + k + j * m_CurrentShape.x;

				int lowerBin = (int)(m_Gradients[index].Orientation / binOrientationDifference);

				double voteMagnitude = m_Params.VoteMagnitudeFn(m_Gradients[index].Magnitude); // change this for other functions
				double upperBinVote = voteMagnitude * (m_Gradients[index].Orientation - binOrientation[lowerBin]) / (binOrientationDifference);

				HOG[lowerBin] += voteMagnitude - upperBinVote;
				if (lowerBin < m_Params.BinCount - 1)
					HOG[lowerBin + 1] += upperBinVote;
			}
		}

		m_Cells.push_back({ HOG });

		if ((i + m_Params.CellSize) % m_CurrentShape.x == 0)
			i += (m_Params.CellSize - 1) * m_CurrentShape.x;

	}

	m_CurrentShape.x /= m_Params.CellSize;
	m_CurrentShape.y /= m_Params.CellSize;
	m_CurrentShape.c = 1;
}

void SingleHOGExecutor::ConstructAndNormalizeBlocks()
{
	m_Blocks.clear();
	m_Blocks.reserve((m_CurrentShape.y - m_Params.BlockSize + 1) * (m_CurrentShape.x - m_Params.BlockSize + 1));
	for (int i = 0; i < m_Cells.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);
		if (coords.x + m_Params.BlockSize - 1 >= m_CurrentShape.x || coords.y + m_Params.BlockSize - 1 >= m_CurrentShape.y)
			continue;

		std::vector<double> HOG(m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize, 0.0);
		for (int j = 0; j < m_Params.BlockSize; j++)
		{
			for (int k = 0; k < m_Params.BlockSize; k++)
			{
				int index = i + k + j * m_CurrentShape.x;
				//std::cout << "index = " << index << ", flat index = " << m_Params.BinCount * k + m_Params.BinCount * m_Params.BlockSize * j << '\n';
				//std::cout << "m_Cells.size() = " << m_Cells.size() << '\n';
				std::copy(m_Cells[index].m_HOG.begin(), m_Cells[index].m_HOG.end(), HOG.begin() + m_Params.BinCount * k + m_Params.BinCount * m_Params.BlockSize * j);
			}
		}

		HOG = m_Params.BlockNormalizationFn(HOG);



		//for (int l = 0; l < HOG.size(); l++)
		//	std::cout << HOG[l] << ", ";



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


SlidingHOGExecutor::SlidingHOGExecutor(Image image, HOGParameter::Parameters params, uint32_t stride)
	:  m_Stride(stride)
{
	m_Params = params;
	m_Stride = stride;
	m_Image = image;
	m_CurrentShape = m_Image.GetShape();
}


SlidingHOGExecutor::SlidingHOGExecutor(const SlidingHOGExecutor& other, double scale)
	: SlidingHOGExecutor(other)
{
	m_Image = Utilities::Resize(other.m_Image, scale * other.m_Image.GetShape().x, scale * other.m_Image.GetShape().y);
	m_CurrentShape = m_Image.GetShape();
}



void SlidingHOGExecutor::Execute()
{
	//m_Image = Utilities::Crop(m_Image, s_Margin, s_Margin, m_CurrentShape.x - (2 * s_Margin), m_CurrentShape.y - (2 * s_Margin));

	ComputeGradients();

	ConstructAndVoteCells();
	ConstructAndNormalizeBlocks();

	//std::vector<HOG> hogs = GetHOGs();


}




void SlidingHOGExecutor::ConstructAndVoteCells()
{
	// create the bins
	std::vector<double> binOrientation = s_CreateOrientationBins(m_Params.SignedOrientationBins, m_Params.BinCount);
	double binOrientationDifference = (m_Params.SignedOrientationBins ? 360 : 180) / (double)(m_Params.BinCount - 1);

	m_Cells.clear();
	m_Cells.reserve((m_CurrentShape.y - m_Params.CellSize + 1) * (m_CurrentShape.x - m_Params.CellSize + 1));

	// recheck the condition of this for loop. 
	for (int i = 0; i < m_Gradients.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);

		//check this if statement
		if (coords.x + m_Params.CellSize > m_CurrentShape.x || coords.y + m_Params.CellSize > m_CurrentShape.y)
			continue;

		std::vector<double> HOG(m_Params.BinCount, 0.0);
		for (int j = 0; j < m_Params.CellSize; j++)
		{
			for (int k = 0; k < m_Params.CellSize; k++)
			{
				int index = i + k + j * m_CurrentShape.x;

				int lowerBin = (int)(m_Gradients[index].Orientation / binOrientationDifference);

				double voteMagnitude = m_Params.VoteMagnitudeFn(m_Gradients[index].Magnitude); // change this for other functions
				double upperBinVote = voteMagnitude * (m_Gradients[index].Orientation - binOrientation[lowerBin]) / (binOrientationDifference);

				HOG[lowerBin] += voteMagnitude - upperBinVote;
				if (lowerBin < m_Params.BinCount - 1)
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
	m_Blocks.reserve((m_CurrentShape.y - m_Params.CellSize * m_Params.BlockSize + 1) * (m_CurrentShape.y - m_Params.CellSize * m_Params.BlockSize + 1));

	for (int i = 0; i < m_Gradients.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);

		//check this if statement
		if (coords.x + m_Params.CellSize * m_Params.BlockSize > m_CurrentShape.x || coords.y + m_Params.CellSize * m_Params.BlockSize > m_CurrentShape.y)
			continue;

		std::vector<double> HOG(m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize, 0.0);
		for (int j = 0; j < m_Params.BlockSize; j++)
		{
			for (int k = 0; k < m_Params.BlockSize; k++)
			{
				// check this index. index in gradients vs index in cells
				int index = i - ((i / (m_CurrentShape.c * m_CurrentShape.x)) * (m_Params.CellSize - 1)) + k * m_Params.CellSize + (m_CurrentShape.x - m_Params.CellSize + 1) * j * m_Params.CellSize;// -((i / (m_CurrentShape.c * m_CurrentShape.x)) * (m_Params.CellSize - 1));
				//std::cout << "index = " << index << ", flat index = " << m_Params.BinCount * k + m_Params.BinCount * m_Params.BlockSize * j << '\n';
				//std::cout << "m_Cells.size() = " << m_Cells.size() << '\n';
				std::copy(m_Cells[index].m_HOG.begin(), m_Cells[index].m_HOG.end(), HOG.begin() + m_Params.BinCount * k + m_Params.BinCount * m_Params.BlockSize * j);
			}
		}

		// normalize
		HOG = m_Params.BlockNormalizationFn(HOG);

		m_Blocks.push_back({ HOG });
	}

	//std::cout << "BLOCKSSIZE = " << m_Blocks.size() << '\n';
}

HOG SlidingHOGExecutor::GetHOG(uint32_t flatIndex)
{
	// flatIndex is coords for the detection window
	// index is converted to bloock coords for the detection window inside margin
	Shape coords = Utilities::FlatToCoords(flatIndex, m_CurrentShape);

	std::cout << "flatIndex = " << flatIndex << '\n';

	//check this if statement
	if (coords.x + m_Params.SlidingWindowSize > m_CurrentShape.x || coords.y + m_Params.SlidingWindowSize > m_CurrentShape.y)
		throw std::runtime_error("SLIDING WINDOW OUTSIDE OF IMAGE");

	// iuhagroiuaeru
	int index = flatIndex - ((flatIndex / (m_CurrentShape.c * m_CurrentShape.x)) * (m_Params.CellSize * m_Params.BlockSize - 1));
	index = index + m_Params.Margin + (m_CurrentShape.x - m_Params.CellSize * m_Params.BlockSize + 1) * m_Params.Margin;

	// hog for current detection window
	HOG hog;
	hog.m_HOG.clear();

	uint64_t rowBlockCount = (((m_Params.SlidingWindowSize - 2 * m_Params.Margin) / m_Params.CellSize) - m_Params.BlockSize + 1);
	hog.m_HOG.resize(rowBlockCount * rowBlockCount * m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize);


	// COLLECT HOG OF CURRENT DETECTION WINDOW

	for (int j = 0; j < rowBlockCount; j++)
	{
		for (int k = 0; k < rowBlockCount; k++)
		{
			int index2 = index + k * m_Params.CellSize + j * m_Params.CellSize * (m_CurrentShape.x - m_Params.CellSize * m_Params.BlockSize + 1);
			//std::cout << "index = " << index2 << ", flat index = " << (m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize) * (k + rowBlockCount * j) << '\n';
			std::copy(m_Blocks[index2].m_HOG.begin(), m_Blocks[index2].m_HOG.end(), hog.m_HOG.begin() + (m_Params.BinCount * m_Params.BlockSize * m_Params.BlockSize) * (k + rowBlockCount * j));
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
	hogs.reserve((m_CurrentShape.y - m_Params.SlidingWindowSize + 1) * (m_CurrentShape.x - m_Params.SlidingWindowSize + 1));


	ResetHOGGetterIndex();
	uint32_t currentIndex = GetCurrentHOGGetterIndex();
	uint32_t count = 0;
	do
	{
		std::cout << "Current Index = " << currentIndex << '\n';
		hogs.push_back(GetHOG(currentIndex));
		AdvanceHOGGetterIndex();
		currentIndex = GetCurrentHOGGetterIndex();
		count++;
	} while (currentIndex != 0);


#if 0
	// i is coords for the detection window
	// index is coords for the detection window inside margin
	for (int i = 0; i < m_Gradients.size(); i += m_Stride)
	{
		Shape coords = Utilities::FlatToCoords(i, m_CurrentShape);
		//check this if statement
		if (coords.x + m_SlidingWindowSize > m_CurrentShape.x)
		{
			i = m_Stride * m_CurrentShape.x + m_CurrentShape.x * (i / m_CurrentShape.x) - m_Stride;
			continue;
		}
		else if (coords.y + m_SlidingWindowSize > m_CurrentShape.y)
		{
			break;
		}
		std::cout << "i = " << i << '\n';
		hogs.push_back(GetHOG(i));
	}
#endif

	std::cout << "window count = " << hogs.size() << '\n';
	std::cout << "hog feature count for each window = " << hogs[0].m_HOG.size();

	return hogs;
}

void SlidingHOGExecutor::AdvanceHOGGetterIndex()
{
	Shape coords = Utilities::FlatToCoords(m_CurrentHOGGetterIndex, m_CurrentShape);
	//check this if statement
	if (coords.y + m_Params.SlidingWindowSize + m_Stride - 1 >= m_CurrentShape.y && coords.x + m_Params.SlidingWindowSize + m_Stride - 1 >= m_CurrentShape.x)
		m_CurrentHOGGetterIndex = 0;
	else if (coords.x + m_Params.SlidingWindowSize + m_Stride - 1 >= m_CurrentShape.x)
		m_CurrentHOGGetterIndex = m_Stride * m_CurrentShape.x + m_CurrentShape.x * (m_CurrentHOGGetterIndex / m_CurrentShape.x);
	else
		m_CurrentHOGGetterIndex += m_Stride;
}


