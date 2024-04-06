#include "pch.h"
#include "framework.h"
#include "Utilities.h"

Random Random::s_Random;

static std::vector<double> FromCVMat(cv::Mat& mat)
{
	uint32_t channels = (mat.type() / 8) + 1;
	std::vector<double> pixels(mat.rows * mat.cols * channels);

	Shape shape = { mat.cols, mat.rows, channels };
	for (int i = 0; i < mat.rows * mat.cols * channels; i++)
	{
		Shape coords = Utilities::FlatToCoords(i, shape);
		pixels[i] = (double)mat.at<cv::Vec3b>(coords.y, coords.x).val[coords.c];
	}

	return pixels;
}

static cv::Mat ToCVMat(std::vector<double>& pixels, const Shape& shape)
{
	int type = CV_MAKETYPE(CV_8U, shape.c);
	cv::Mat mat(shape.y, shape.x, type);

	for (int i = 0; i < pixels.size(); i++)
	{
		Shape coords = Utilities::FlatToCoords(i, shape);
		mat.at<cv::Vec3b>(coords.y, coords.x).val[coords.c] = (uint8_t)pixels[i];
	}

	return mat;
}


static std::vector<double> MultMatVec(std::vector<std::vector<double>> mat, std::vector<double> vec)
{
	if (mat[0].size() != vec.size())
		throw std::runtime_error("UNMATCHING MATRIX AND VECTOR SIZE FOR MULTIPLICATION!");

	std::vector<double> result = vec;

	for (int i = 0; i < result.size(); i++)
	{
		double sum = 0.0;
		for (int j = 0; j < result.size(); j++)
			sum += mat[i][j] * vec[j];
		result[i] = sum;
	}

	return result;
}

static double MultVecVec(std::vector<double> vec1, std::vector<double> vec2)
{
	if (vec1.size() != vec2.size())
		throw std::runtime_error("UNMATCHING VECTOR AND VECTOR SIZE FOR MULTIPLICATION!");

	double sum = 0.0;

	for (int i = 0; i < vec1.size(); i++)
		sum += vec1[i] * vec2[i];

	return sum;
}

// p is the function value, i.e. y.
static double CubicInterpolation(double p0, double p1, double p2, double p3, double x)
{
	return
		(-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * x * x * x
		+ (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3) * x * x
		+ (-0.5 * p0 + 0.5 * p2) * x + p1;
}


namespace Utilities
{
	const double s_Epsilon = 0.00001;

	void DrawFlush()
	{
		cv::waitKey(0);
	}

	std::string ParseName(std::string path)
	{
		std::string name = path;

		size_t pos = path.find_last_of('/');
		if (pos != std::string::npos)
			name = name.substr(pos, std::string::npos);

		pos = path.find_last_of('.');
		if (pos != std::string::npos)
			name = name.substr(0, name.length() - pos + 1);

		return name;
	}

	Image Resize(Image image, uint32_t targetX, uint32_t targetY)
	{
		Shape sourceShape = image.GetShape();
		Shape targetShape = { targetX, targetY, sourceShape.c };

		std::vector<double> pixels = image.GetPixels();
		std::vector<double> pixelsResult(targetX * targetY * sourceShape.c, 0.0);

		for (int i = 0; i < pixelsResult.size(); i++)
		{
			Shape resultCoords = FlatToCoords(i, targetShape);
			std::vector<double> sourceCoords =
			{
				resultCoords.x * sourceShape.x / (double)targetShape.x,
				resultCoords.y * sourceShape.y / (double)targetShape.y,
				(double)(i % sourceShape.c)
			};

			double xOffset = sourceCoords[0] - (int)sourceCoords[0];
			double yOffset = sourceCoords[1] - (int)sourceCoords[1];

			std::vector<double> p;

			std::vector<double> yAnchors;
			yAnchors.reserve(4);

			for (int k = -1; k < 3; k++)
			{
				p.clear();
				p.reserve(4);
				// find y anchor
				for (int l = -1; l < 3; l++)
				{
					if (sourceCoords[0] + l >= sourceShape.x || sourceCoords[0] + l < 0 || sourceCoords[1] + k >= sourceShape.y || sourceCoords[1] + k < 0)
						p.push_back(0);
					else
						p.push_back(std::max(0.0, std::min(255.0, pixels[CoordsToFlat(sourceCoords[0] + l, sourceCoords[1] + k, sourceCoords[2], sourceShape)])));
				}

				yAnchors.push_back(CubicInterpolation(p[0], p[1], p[2], p[3], xOffset));

			}

			pixelsResult[i] = std::max(0.0, std::min(255.0, CubicInterpolation(yAnchors[0], yAnchors[1], yAnchors[2], yAnchors[3], yOffset)));


		}

		return Image(pixelsResult, targetShape, image.GetName());
	}

	Image GaussianBlur(Image image, double sigma, uint32_t kernel_size)
	{
		std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size, 0.0));

		double temp = 1.0 / (2 * std::numbers::pi * sigma * sigma);
		double temp2 = std::exp(-1.0 / (2 * sigma * sigma));

		for (int i = 0; i < kernel_size; i++)
		{
			for (int j = 0; j < kernel_size; j++)
			{
				int x = i - (kernel_size / 2);
				int y = j - (kernel_size / 2);
				kernel[i][j] = temp * std::pow(temp2, (x * x + y * y));
			}
		}

		return Convolve(image, kernel);
	}


	Image Rotate(Image image, std::vector<double> degrees)
	{
		std::vector<double> pixels = image.GetPixels();
		Shape shape = image.GetShape();

		std::vector<double> pixelsResult(pixels.size(), 0);

		int xCenter = shape.x / 2;
		int yCenter = shape.y / 2;

		double rad = degrees[Random::RandomNumber(0, degrees.size() - 1)] * (std::numbers::pi / 180.0);

		std::vector<std::vector<double>> mat =
		{
			{std::cos(rad), -1 * std::sin(rad)},
			{std::sin(rad), std::cos(rad)}
		};


		for (int i = 0; i < pixelsResult.size(); i++)
		{
			Shape xyc = FlatToCoords(i, shape);

			std::vector<double> point = { (double)((int)xyc.x - xCenter), (double)((int)xyc.y - yCenter) };
			point = MultMatVec(mat, point);
			std::vector<int> pointi = { (int)(point[0] + xCenter), (int)(point[1] + yCenter) };

			if (pointi[0] < 0 || pointi[0] >= shape.x || pointi[1] < 0 || pointi[1] >= shape.y)
				continue;

			pixelsResult[i] = pixels[CoordsToFlat(pointi[0], pointi[1], xyc.c, shape)];
		}

		return Image(pixelsResult, shape, image.GetName());
	}

	Image HorizontalFlip(Image image, double probability)
	{
		if (Random::RandomProbability() > probability)
			return image;

		Shape shape = image.GetShape();
		std::vector<double> pixels = image.GetPixels();
		std::vector<double> pixelsResult = pixels;

		for (int i = 0; i < pixels.size(); i++)
			pixelsResult[i] = pixels[i + shape.c * 2 * (((double)(shape.x - 1) / 2) - (i / shape.c) % shape.x)];

		return Image(pixelsResult, shape, image.GetName());
	}

	Image Crop(Image image, uint32_t xMin, uint32_t yMin, uint32_t xSize, uint32_t ySize)
	{
		Shape shape = image.GetShape();
		Shape targetShape = { xSize, ySize, shape.c };

		if (xMin + xSize > shape.x || yMin + ySize > shape.y)
			throw std::runtime_error("CROPPING TOO BIG!");

		std::vector<double> pixels = image.GetPixels();
		std::vector<double> pixelsResult(xSize * ySize * shape.c);

		for (int i = 0; i < pixelsResult.size(); i++)
		{
			Shape coords = FlatToCoords(i, targetShape);
			coords.x += xMin;
			coords.y += yMin;
			pixelsResult[i] = pixels[CoordsToFlat(coords.x, coords.y, coords.c, shape)];
		}

		return Image(pixelsResult, targetShape, image.GetName());
	}


	Image Convolve(Image image, std::vector<std::vector<double>> kernel)
	{
		Image temp = image;
		Shape shape = image.GetShape();
		for (int i = 0; i < shape.y; i++)
		{
			for (int j = 0; j < shape.x; j++)
			{
				for (int k = 0; k < shape.c; k++)
				{
					double sum = 0.0;
					for (int y = 0; y < kernel.size(); y++)
					{
						for (int x = 0; x < kernel.size(); x++)
						{
							int xc = j + x - (kernel.size() / 2);
							int yc = i + y - (kernel.size() / 2);
							if (!(xc > 0 && yc > 0 && xc < shape.x && yc < shape.y))
								continue;
							sum += image.GetPixel(xc, yc, k) * kernel[y][x];
						}
					}
					temp.SetPixel(j, i, k, sum);
				}
			}
		}

		return temp;
	}

	std::vector<double> Convolve(std::vector<double> image, std::vector<std::vector<double>> kernel, Shape shape)
	{
		std::vector<double> temp = image;
		for (int i = 0; i < shape.y; i++)
		{
			for (int j = 0; j < shape.x; j++)
			{
				for (int k = 0; k < shape.c; k++)
				{
					double sum = 0.0;
					for (int y = 0; y < kernel.size(); y++)
					{
						for (int x = 0; x < kernel.size(); x++)
						{
							int xc = j + x - (kernel.size() / 2);
							int yc = i + y - (kernel.size() / 2);
							if (!(xc > 0 && yc > 0 && xc < shape.x && yc < shape.y))
								continue;
							sum += image[Utilities::CoordsToFlat(xc, yc, k, shape)] * kernel[y][x];
						}
					}
					temp[Utilities::CoordsToFlat(j, i, k, shape)] = sum;
				}
			}
		}

		return temp;
	}

	std::vector<std::string> GetFilesInDirectory(std::string directoryPath)
	{
		std::string absPath = std::filesystem::current_path().string() + "\\" + directoryPath;
		std::cout << absPath << '\n';
		std::vector<std::string> paths;

		for (const auto& entry : std::filesystem::directory_iterator(absPath))
			paths.push_back(entry.path().string());

		return paths;
	}
}




Image Image::LoadImage(std::string path)
{
	cv::Mat imageMat = cv::imread(path, cv::IMREAD_COLOR);
	std::string name = Utilities::ParseName(path);

	return Image(imageMat, name);
}

std::vector<Image> Image::LoadImages(std::vector<std::string> paths)
{
	std::vector<Image> images;
	images.reserve(paths.size());

	for (int i = 0; i < paths.size(); i++)
	{
		auto temp = LoadImage(paths[i]);
		images.push_back(temp);
	}

	return images;
}

void Image::Save(std::string path)
{
	m_CVMat = ToCVMat(m_Pixels, m_Shape);
	cv::imwrite(path, m_CVMat);
}

Image::Image()
	: m_CVMat(cv::Mat(1, 1, 1)) {}

//Image::Image(Image& other)
//	: m_Pixels(other.m_Pixels), m_CVMat(other.m_CVMat.clone()), m_Name(other.m_Name), m_Shape(other.m_Shape) {}

Image::Image(cv::Mat image, std::vector<double> pixels, std::string name)
	: m_CVMat(image.clone()), m_Pixels(pixels), m_Name(name), m_Shape({(uint32_t)image.cols, (uint32_t)image.rows, (uint32_t)(image.type() / 8) + 1 }) {}

Image::Image(cv::Mat image, std::string name)
	: m_CVMat(image.clone()), m_Pixels(FromCVMat(image)), m_Name(name), m_Shape({ (uint32_t)image.cols, (uint32_t)image.rows, (uint32_t)(image.type() / 8) + 1 }) {}

Image::Image(std::vector<double> pixels, Shape shape, std::string name)
	: m_CVMat(ToCVMat(pixels, shape)), m_Pixels(pixels), m_Name(name), m_Shape(shape) {}


void Image::Draw()
{
	m_CVMat = ToCVMat(m_Pixels, m_Shape);
	cv::imshow(m_Name, m_CVMat);
}



uint64_t Random::RandomNumber(uint64_t min, uint64_t max)
{
	uint64_t temp = s_Random.m_RNG();
	if (temp == UINT64_MAX)
		return min;
	return temp * (double)((int64_t)(max + 1) - min) / UINT64_MAX + min;
}

double Random::RandomProbability()
{
	return (double)s_Random.m_RNG() / UINT64_MAX;
}