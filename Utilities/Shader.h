#pragma once

namespace Shader
{
	// Compute Shader Program Index
	enum class CSPIndex
	{
		Torso = 0,
		Head,
		Hand,
		Leg
	};

	// Compute Shader Program
	extern std::vector<unsigned int> CSP;

	// Shader Storage Buffer Object Index
	enum class SSBOIndex
	{
		ModelTorso = 0,
		ModelHead,
		ModelHand,
		ModelLeg,

		InputTorso,
		InputHead,
		InputHand,
		InputLeg,
		
		OutputTorso,
		OutputHead,
		OutputHand,
		OutputLeg,
	};

	// Shader Storage Buffer Object
	extern std::vector<unsigned int> SSBOModel;
	extern std::vector<unsigned int> SSBOOutput;

	void Initialize(std::vector<std::vector<double>>& modelTorsoData,
		std::vector<std::vector<double>>& modelHeadData,
		std::vector<std::vector<double>>& modelHandData,
		std::vector<std::vector<double>>& modelLegData);

	unsigned int CreateProgram(std::string shaderSourcePath);
	unsigned int CreateComputeShader(std::string shaderSourcePath);


	// SSBO for output, size is 8 bytes * size
	unsigned int CreateSSBO(SSBOIndex si, uint32_t size);

	// SSBO for input
	unsigned int CreateSSBO(const std::vector<double>& x, SSBOIndex si);

	// SSBO for model
	unsigned int CreateSSBO(const std::vector<std::vector<double>>& x, SSBOIndex si);

	std::vector<double> DispatchComputeShader(unsigned int programID, SSBOIndex inputSSBOIndex, const std::vector<double>& inputData, uint32_t size);


}