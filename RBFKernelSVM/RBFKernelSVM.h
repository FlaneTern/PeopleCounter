#pragma once

struct Dataset
{
	std::vector<std::vector<double>> Features;
	std::vector<double> Labels;
};


class RBFKernelSVM
{
public:
	RBFKernelSVM(double kernelParameterGamma);
	RBFKernelSVM(double kernelParameterGamma, std::string name);
	void Train(const Dataset& data, double regularizationParameterC, uint32_t maxIterations, double endDeltaThreshold, double learningRate);
	double Predict(std::vector<double> features);

	void Save(std::string path);

	inline std::string GetName() { return m_Name; }
	inline std::string SetName(const std::string& name) { m_Name = name; }

	static RBFKernelSVM Load(std::string path);
	RBFKernelSVM(std::string name, double kernelParameterGamma, double bias, std::vector<double> lagrangeMultipliers, Dataset dataset);

private:
	std::string m_Name;
	double m_KernelParameterGamma;
	double m_Bias;
	std::vector<double> m_LagrangeMultipliers;
	Dataset m_Dataset;


};