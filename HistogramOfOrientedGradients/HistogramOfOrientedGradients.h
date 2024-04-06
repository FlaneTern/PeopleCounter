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
	virtual HOG Execute();

protected:
	void ComputeGradients(const std::vector<std::vector<double>>& kernelX, const std::vector<std::vector<double>>& kernelY);
	virtual void ConstructAndVoteCells() = 0;
	virtual void ConstructAndNormalizeBlocks() = 0;
	virtual HOG GetHOG();

	std::vector<Gradient> m_Gradients;


	Image m_Image;
	std::vector<HOG> m_Cells;
	std::vector<HOG> m_Blocks;

	uint32_t m_CellSize;
	uint32_t m_BlockSize;


	Shape m_CurrentShape;

	static constexpr uint32_t s_Margin = 16;
};

class SingleHOGExecutor : public HOGExecutor
{
public:
	SingleHOGExecutor(Image image);
	Image ToImage();
	virtual HOG Execute() override;
private:
	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
	virtual HOG GetHOG() override;
};

class SlidingHOGExecutor : public HOGExecutor
{
	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
};