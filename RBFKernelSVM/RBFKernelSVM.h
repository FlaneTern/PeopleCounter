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
	double Predict(std::vector<double> features) const;
	double PredictShader(std::vector<double> features) const;
	double PredictShader(std::vector<double> features, unsigned int programID, unsigned int inputSSBOIndex) const;

	void Save(std::string path);

	inline std::string GetName() { return m_Name; }
	inline std::string SetName(const std::string& name) { m_Name = name; }

	// DONT MODIFY OUTSIDE OF THIS CLASS
	inline std::vector<std::vector<double>> GetDataFeatures() { return m_Dataset.Features; }

	static RBFKernelSVM Load(std::string path);
	RBFKernelSVM(double kernelParameterGamma, double bias, std::vector<double> lagrangeMultipliers, Dataset dataset, std::string name);

	void Log();

	inline void SetShader(unsigned int programID, unsigned int inputSSBOIndex)
	{
		m_ProgramID = programID;
		m_InputSSBOIndex = inputSSBOIndex;
	}

	std::vector<double> m_LagrangeMultipliers;
private:
	std::string m_Name;
	double m_KernelParameterGamma;
	double m_Bias;
	//std::vector<double> m_LagrangeMultipliers;
	Dataset m_Dataset;

	// shader stuff
	unsigned int m_ProgramID;
	unsigned int m_InputSSBOIndex;
};