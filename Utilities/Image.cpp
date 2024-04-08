#include "pch.h"
#include "Utilities.h"



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
	m_CVMat = Utilities::ToCVMat(m_Pixels, m_Shape);
	cv::imwrite(path, m_CVMat);
}

Image::Image()
	: m_CVMat(cv::Mat(1, 1, 1)) {}

//Image::Image(Image& other)
//	: m_Pixels(other.m_Pixels), m_CVMat(other.m_CVMat.clone()), m_Name(other.m_Name), m_Shape(other.m_Shape) {}

Image::Image(cv::Mat image, std::vector<double> pixels, std::string name)
	: m_CVMat(image.clone()), m_Pixels(pixels), m_Name(name), m_Shape({ (uint32_t)image.cols, (uint32_t)image.rows, (uint32_t)(image.type() / 8) + 1 }) {}

Image::Image(cv::Mat image, std::string name)
	: m_CVMat(image.clone()), m_Pixels(Utilities::FromCVMat(image)), m_Name(name), m_Shape({ (uint32_t)image.cols, (uint32_t)image.rows, (uint32_t)(image.type() / 8) + 1 }) {}

Image::Image(std::vector<double> pixels, Shape shape, std::string name)
	: m_CVMat(Utilities::ToCVMat(pixels, shape)), m_Pixels(pixels), m_Name(name), m_Shape(shape) {}


void Image::Draw()
{
	m_CVMat = Utilities::ToCVMat(m_Pixels, m_Shape);
	cv::imshow(m_Name, m_CVMat);
}
