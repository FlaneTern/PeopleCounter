#pragma once

struct Gradient
{
	double Magnitude;
	double Orientation;
};

namespace Kernel
{
	extern const std::vector<std::vector<double>> s_SobelX;
	extern const std::vector<std::vector<double>> s_SobelY;

	extern const std::vector<std::vector<double>> s_FDX;
	extern const std::vector<std::vector<double>> s_FDY;
}

struct HOG
{
	HOG();
	HOG(std::vector<double> hog);
	std::vector<double> m_HOG;
};

class HOGExecutor
{
public:

protected:
	void ComputeGradients(const std::vector<std::vector<double>>& kernelX, const std::vector<std::vector<double>>& kernelY);
	virtual void ConstructAndVoteCells() = 0;
	virtual void ConstructAndNormalizeBlocks() = 0;

	std::vector<Gradient> m_Gradients;


	Image m_Image;
	std::vector<HOG> m_Cells;
	std::vector<HOG> m_Blocks;

	uint32_t m_CellSize;
	uint32_t m_BlockSize;

	Shape m_CurrentShape;

	static constexpr uint32_t s_Margin = 16;
	static constexpr uint32_t s_SlidingWindowSize = 128;
};

// SingleHOGExecutor works on cell coordinates after cells are constructed
class SingleHOGExecutor : public HOGExecutor
{
public:
	SingleHOGExecutor(Image image);
	Image ToImage();
	HOG Execute();

private:

	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
	HOG GetHOG();
};

// SlidingHOGExecutor works on pixel coordinates, trimmed by cells and then by blocks
class SlidingHOGExecutor : public HOGExecutor
{
public:
	SlidingHOGExecutor(Image image);
	std::vector<HOG> Execute();
private:
	
	Shape m_CellShape;
	Shape m_BlockShape;

	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
	std::vector<HOG> GetHOGs();
};