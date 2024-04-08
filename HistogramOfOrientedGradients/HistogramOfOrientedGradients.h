#pragma once

struct Gradient
{
	double Magnitude;
	double Orientation;
};

namespace HOGParameter
{
	typedef std::vector<std::vector<double>> Kernel;
	extern const Kernel s_SobelX;
	extern const Kernel s_SobelY;

	extern const Kernel s_FDX;
	extern const Kernel s_FDY;

	typedef std::function<double(double)> VoteMagnitudeFn;
	typedef std::function<std::vector<double>(std::vector<double>)> BlockNormalizationFn;

	double s_Linear(double magnitude);
	double s_Square(double magnitude);
	double s_SquareRoot(double magnitude);

	std::vector<double> s_L1Norm(std::vector<double> hog);
	std::vector<double> s_L2Norm(std::vector<double> hog);

	struct Parameters
	{
		uint32_t SlidingWindowSize;
		uint32_t Margin;
		HOGParameter::Kernel XKernel;
		HOGParameter::Kernel YKernel;
		uint32_t CellSize;
		uint32_t BlockSize;
		bool SignedOrientationBins;
		uint32_t BinCount;
		HOGParameter::VoteMagnitudeFn VoteMagnitudeFn;
		HOGParameter::BlockNormalizationFn BlockNormalizationFn;
	};
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

	inline Shape GetShape() const { return m_CurrentShape; }
	inline HOGParameter::Parameters GetParams() const { return m_Params; }


protected:
	void ComputeGradients();
	virtual void ConstructAndVoteCells() = 0;
	virtual void ConstructAndNormalizeBlocks() = 0;

	Image m_Image;

	HOGParameter::Parameters m_Params;

	std::vector<Gradient> m_Gradients;
	Shape m_CurrentShape;

	std::vector<HOG> m_Blocks;
	std::vector<HOG> m_Cells;
};

// SingleHOGExecutor works on cell coordinates after cells are constructed
class SingleHOGExecutor : public HOGExecutor
{
public:
	SingleHOGExecutor(Image image, HOGParameter::Parameters params);

	void Execute();

	Image ToImage();
	HOG GetHOG();

private:

	virtual void ConstructAndVoteCells() override;
	virtual void ConstructAndNormalizeBlocks() override;
};

// SlidingHOGExecutor works on pixel coordinates, trimmed by cells and then by blocks
class SlidingHOGExecutor : public HOGExecutor
{
public:
	SlidingHOGExecutor(Image image, HOGParameter::Parameters params, uint32_t stride);
	SlidingHOGExecutor(const SlidingHOGExecutor& other, double scale);

	void Execute();

	inline uint32_t GetCurrentHOGGetterIndex() const { return m_CurrentHOGGetterIndex; }
	inline Shape GetCurrentHOGGetterIndexCoords() const { return Utilities::FlatToCoords(m_CurrentHOGGetterIndex, m_CurrentShape); }
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