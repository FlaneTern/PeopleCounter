#pragma once

struct Shape
{
	uint32_t x;
	uint32_t y;
	uint32_t c;
};

namespace Utilities
{
	extern const double s_Epsilon;

	inline uint32_t CoordsToFlat(uint32_t x, uint32_t y, uint32_t c, const Shape& shape)
		{ return y * shape.x * shape.c + x * shape.c + c; }

	inline Shape FlatToCoords(uint32_t i, const Shape& shape) 
		{ return { (i / shape.c) % shape.x, i / (shape.c * shape.x), i % shape.c }; }
}



class Random
{
public:

	static uint64_t RandomNumber(uint64_t min, uint64_t max);
	static double RandomProbability();

	static Random s_Random;
private:
	std::mt19937_64 m_RNG = std::mt19937_64(std::chrono::high_resolution_clock::now().time_since_epoch().count());
};


class Image
{
public:
	Image();
	//Image(Image& other);
	Image(cv::Mat image, std::vector<double> pixels, std::string name);
	Image(cv::Mat image, std::string name);
	Image(std::vector<double> pixels, Shape shape, std::string name);

	void Draw();

	inline double GetPixel(uint32_t x, uint32_t y, uint32_t c) const { return m_Pixels[Utilities::CoordsToFlat(x, y, c, m_Shape)]; }
	inline void SetPixel(uint32_t x, uint32_t y, uint32_t c, double value) { m_Pixels[Utilities::CoordsToFlat(x, y, c, m_Shape)] = value; }
	inline std::vector<double> GetPixels() { return m_Pixels; }

	inline Shape GetShape() const { return m_Shape; }
	inline std::string GetName() const { return m_Name; }

	void Save(std::string path);

	static Image LoadImage(std::string path);
	static std::vector<Image> LoadImages(std::vector<std::string> paths);

private:
	std::vector<double> m_Pixels;
	cv::Mat m_CVMat;
	std::string m_Name;
	Shape m_Shape;
};




namespace Utilities
{
	void DrawFlush();

	std::string ParseName(std::string path);

	std::vector<std::string> GetFilesInDirectory(std::string directoryPath);

	Image Convolve(Image image, std::vector<std::vector<double>> kernel);
	std::vector<double> Convolve(std::vector<double> image, std::vector<std::vector<double>> kernel, Shape shape);


	Image Resize(Image image, uint32_t targetX, uint32_t targetY);
	Image GaussianBlur(Image image, double sigma, uint32_t kernel_size);
	Image Rotate(Image image, std::vector<double> degrees);
	Image HorizontalFlip(Image image, double probability);
	Image Crop(Image image, uint32_t xMin, uint32_t yMin, uint32_t xSize, uint32_t ySize);

	std::vector<double> FromCVMat(cv::Mat& mat);
	cv::Mat ToCVMat(std::vector<double>& pixels, const Shape& shape);
	std::vector<double> MultMatVec(std::vector<std::vector<double>> mat, std::vector<double> vec);
	double MultVecVec(std::vector<double> vec1, std::vector<double> vec2);

	// p is the function value, i.e. y.
	double CubicInterpolation(double p0, double p1, double p2, double p3, double x);

}

