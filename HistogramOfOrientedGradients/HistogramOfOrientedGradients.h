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
	void Execute();
	HOG GetHOG();

private:

	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
};

// SlidingHOGExecutor works on pixel coordinates, trimmed by cells and then by blocks
class SlidingHOGExecutor : public HOGExecutor
{
public:
	SlidingHOGExecutor(Image image, uint32_t stride);
	void Execute();

	inline uint32_t GetCurrentHOGGetterIndex() const { return m_CurrentHOGGetterIndex; }
	inline void ResetHOGGetterIndex() { m_CurrentHOGGetterIndex = 0; }
	void AdvanceHOGGetterIndex();
	HOG GetHOG();
	HOG GetHOG(uint32_t flatIndex);

	// dont use this method, it's gonna blow ur ram up (=
	// use GetHOG with HOGGetterIndex instead
	std::vector<HOG> GetHOGs();

private:
	
	uint32_t m_Stride;

	// m_CurrentHOGGetterIndex is coords for the detection window, works on image/gradient coordinates
	uint32_t m_CurrentHOGGetterIndex;

	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
};