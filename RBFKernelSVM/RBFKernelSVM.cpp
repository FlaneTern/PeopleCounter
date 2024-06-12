#include "pch.h"
#include "RBFKernelSVM.h"

#include "Utilities.h"
#include "Shader.h"

static double RBFKernel(const std::vector<double>& x1, const std::vector<double>& x2, double gamma)
{
	if (x1.size() != x2.size())
		throw std::runtime_error("DIFFERENT X1 AND X2 SIZE ON RBF KERNEL!");

	double sum = 0.0;
	double temp = 0.0;
	for (int i = 0; i < x1.size(); i++)
	{
		temp = x1[i] - x2[i];
		sum += temp * temp;
	}
	sum = std::exp(-1 * gamma * sum);

	return sum;
}

RBFKernelSVM::RBFKernelSVM(double kernelParameterGamma)
	: m_KernelParameterGamma(kernelParameterGamma), m_Bias(0.0) {}

RBFKernelSVM::RBFKernelSVM(double kernelParameterGamma, std::string name)
	: m_KernelParameterGamma(kernelParameterGamma), m_Bias(0.0), m_Name(name) {}

RBFKernelSVM::RBFKernelSVM(double kernelParameterGamma, double bias, std::vector<double> lagrangeMultipliers, Dataset dataset, std::string name)
	: m_Name(name), m_KernelParameterGamma(kernelParameterGamma), m_Bias(bias), m_LagrangeMultipliers(lagrangeMultipliers), m_Dataset(dataset) {}

RBFKernelSVM RBFKernelSVM::Load(std::string path)
{
	std::string temp = Utilities::ReadFileBinary(path);

	uint64_t currentPos = temp.find('\0');

	std::string name;
	name = temp.substr(0, currentPos);
	currentPos++;

	double kernelParameterGamma;
	double bias;
	uint64_t lagrangeMultipliersCount;
	uint64_t featureCount;

	kernelParameterGamma = *(double*)&temp[currentPos];
	currentPos += sizeof(kernelParameterGamma);

	bias = *(double*)&temp[currentPos];
	currentPos += sizeof(bias);

	lagrangeMultipliersCount = *(uint64_t*)&temp[currentPos];
	currentPos += sizeof(lagrangeMultipliersCount);

	featureCount = *(uint64_t*)&temp[currentPos];
	currentPos += sizeof(featureCount);

	std::vector<double> lagrangeMultipliers(lagrangeMultipliersCount);
	std::vector<double> labels(lagrangeMultipliersCount);
	std::vector<std::vector<double>> features(lagrangeMultipliersCount, std::vector<double>(featureCount, 0.0));

	memcpy(lagrangeMultipliers.data(), &temp[currentPos], sizeof(lagrangeMultipliers[0]) * lagrangeMultipliersCount);
	currentPos += sizeof(lagrangeMultipliers[0]) * lagrangeMultipliersCount;

	memcpy(labels.data(), &temp[currentPos], sizeof(labels[0]) * lagrangeMultipliersCount);
	currentPos += sizeof(labels[0]) * lagrangeMultipliersCount;

	for (int i = 0; i < lagrangeMultipliersCount; i++)
	{
		memcpy(features[i].data(), &temp[currentPos], sizeof(features[i][0]) * featureCount);
		currentPos += sizeof(features[i][0]) * featureCount;
	}
	
	Dataset data = { features, labels };

	return RBFKernelSVM(kernelParameterGamma, bias, lagrangeMultipliers, data, name);
}

void RBFKernelSVM::Save(std::string path)
{
	std::ofstream stream(path, std::ios::binary);

	uint64_t lmSize = (uint64_t)m_LagrangeMultipliers.size();
	uint64_t fsize = (uint64_t)m_Dataset.Features[0].size();

	stream << m_Name << '\0';

	stream.write((char*)&m_KernelParameterGamma, sizeof(m_KernelParameterGamma));
	stream.write((char*)&m_Bias, sizeof(m_Bias));
	stream.write((char*)&lmSize, sizeof(lmSize));
	stream.write((char*)&fsize, sizeof(fsize));

	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		stream.write((char*)&m_LagrangeMultipliers[i], sizeof(m_LagrangeMultipliers[i]));

	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		stream.write((char*)&m_Dataset.Labels[i], sizeof(m_Dataset.Labels[i]));

	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		for (int j = 0; j < m_Dataset.Features[0].size(); j++)
			stream.write((char*)&m_Dataset.Features[i][j], sizeof(m_Dataset.Features[i][j]));

}

void RBFKernelSVM::Train(const Dataset& data, double regularizationParameterC, uint32_t maxIterations, double endDeltaThreshold, double learningRate)
{
	std::cout << "Starting Training";

	m_Dataset = data;

	m_LagrangeMultipliers = std::vector<double>(m_Dataset.Features.size(), 0.0);

	double beta = 1.0;
	bool done = false;

	for (int i = 0; i < maxIterations && !done; i++)
	{
		std::cout << "Training Round " << i << '\n';
		done = true;

		for (int j = 0; j < m_LagrangeMultipliers.size(); j++)
		{
			double temp = 0.0;
			double temp2 = 0.0;
			for (int k = 0; k < m_LagrangeMultipliers.size(); k++)
			{
				double temp3 = m_LagrangeMultipliers[k] * m_Dataset.Labels[j] * m_Dataset.Labels[k];
				temp += temp3;
				temp2 += temp3 * RBFKernel(m_Dataset.Features[j], m_Dataset.Features[k], m_KernelParameterGamma);
			}

			double delta = 1.0 - temp2 - beta * temp;

			m_LagrangeMultipliers[j] += learningRate * delta;
			//std::cout << "Updating Lagrange Multiplier " << j << " = " << m_LagrangeMultipliers[j] << "\t, Delta = " << learningRate * delta << '\n';

			if (m_LagrangeMultipliers[j] < 0.0)
				m_LagrangeMultipliers[j] = 0.0;
			else if (m_LagrangeMultipliers[j] > regularizationParameterC)
				m_LagrangeMultipliers[j] = regularizationParameterC;
			else if (std::abs(delta) > endDeltaThreshold)
				done = false;
		}

		double temp = 0.0;
		for (int j = 0; j < m_LagrangeMultipliers.size(); j++)
			temp += m_LagrangeMultipliers[j] * m_Dataset.Labels[j];

		beta += temp * temp / 2.0;
	}

	int onMarginCount = 0;
	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
	{
		if (m_LagrangeMultipliers[i] <= 0.0 || m_LagrangeMultipliers[i] >= regularizationParameterC)
			continue;

		onMarginCount++;	

		m_Bias += m_Dataset.Labels[i];

		for (int j = 0; j < m_LagrangeMultipliers.size(); j++)
			m_Bias -= m_LagrangeMultipliers[j] * m_Dataset.Labels[j] * RBFKernel(m_Dataset.Features[i], m_Dataset.Features[j], m_KernelParameterGamma);
	}

	if (onMarginCount == 0)
		onMarginCount++;
	m_Bias /= (double)onMarginCount;

	std::cout << "Finished Training";
	
}

double RBFKernelSVM::Predict(std::vector<double> features) const
{
	double result = m_Bias;

	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		result += m_LagrangeMultipliers[i] * m_Dataset.Labels[i] * RBFKernel(m_Dataset.Features[i], features, m_KernelParameterGamma);

	return result;
}

double RBFKernelSVM::PredictShader(std::vector<double> features) const
{
	double result = m_Bias;

	std::vector<double> diff = Shader::DispatchComputeShader(m_ProgramID, (Shader::SSBOIndex)m_InputSSBOIndex, features, m_LagrangeMultipliers.size());

	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		result += m_LagrangeMultipliers[i] * m_Dataset.Labels[i] * std::exp(-1 * m_KernelParameterGamma * diff[i]);

	return result;
}

double RBFKernelSVM::PredictShader(std::vector<double> features, unsigned int programID, unsigned int inputSSBOIndex) const
{
	double result = m_Bias;

	std::vector<double> diff = Shader::DispatchComputeShader(programID, (Shader::SSBOIndex)inputSSBOIndex, features, m_LagrangeMultipliers.size());

	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		result += m_LagrangeMultipliers[i] * m_Dataset.Labels[i] * std::exp(-1 * m_KernelParameterGamma * diff[i]);

	return result;
}

void RBFKernelSVM::Log()
{
	std::cout << "Name = " << m_Name << '\n';
	std::cout << "Kernel Parameter Gamma = " << m_KernelParameterGamma << '\n';
	std::cout << "Bias = " << m_Bias << '\n';

	std::cout << "Lagrange Multipliers :\n";
	for (int i = 0; i < m_LagrangeMultipliers.size(); i++)
		std::cout << "\tLagrange Multiplier " << i << " = " << m_LagrangeMultipliers[i] << '\n';

	//std::cout << "Dataset X :\n";
	//for (int i = 0; i < 1; i++)
	//{
	//	std::cout << "\tX " << i << " = { ";
	//	for (int j = 0; j < m_Dataset.Features[i].size(); j++)
	//		std::cout << m_Dataset.Features[i][j] << ", ";
	//	std::cout << " }\n";
	//}

	std::cout << "Dataset Y :\n";
	for (int i = 0; i < m_Dataset.Labels.size(); i++)
		std::cout << "\tDataset Y = " << m_Dataset.Labels[i] << '\n';

}

