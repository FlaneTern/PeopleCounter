#include "pch.h"

#include "glad/include/glad/glad.h"
#include "glfw-3.4/include/GLFW/glfw3.h"
#include "Shader.h"

#include "Utilities.h"

namespace Shader
{
    std::vector<unsigned int> SSBOModel;
    std::vector<unsigned int> SSBOOutput;
    std::vector<unsigned int> CSP;


    void Initialize(std::vector<std::vector<double>>& modelTorsoData,
        std::vector<std::vector<double>>& modelHeadData,
        std::vector<std::vector<double>>& modelHandData,
        std::vector<std::vector<double>>& modelLegData)
	{
		glfwInit();

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);

        if (window == NULL)
            throw std::runtime_error("wkwkwk");

        glfwMakeContextCurrent(window);

        gladLoadGL();


        CSP.push_back(CreateProgram("ComputeShaderTorso.glsl"));
        SSBOModel.push_back(CreateSSBO(modelTorsoData, SSBOIndex::ModelTorso));
        SSBOOutput.push_back(CreateSSBO(SSBOIndex::OutputTorso, modelTorsoData.size()));

        CSP.push_back(CreateProgram("ComputeShaderHead.glsl"));
        SSBOModel.push_back(CreateSSBO(modelHeadData, SSBOIndex::ModelHead));
        SSBOOutput.push_back(CreateSSBO(SSBOIndex::OutputHead, modelHeadData.size()));

        CSP.push_back(CreateProgram("ComputeShaderHand.glsl"));
        SSBOModel.push_back(CreateSSBO(modelHandData, SSBOIndex::ModelHand));
        SSBOOutput.push_back(CreateSSBO(SSBOIndex::OutputHand, modelHandData.size()));

        CSP.push_back(CreateProgram("ComputeShaderLeg.glsl"));
        SSBOModel.push_back(CreateSSBO(modelLegData, SSBOIndex::ModelLeg));
        SSBOOutput.push_back(CreateSSBO(SSBOIndex::OutputLeg, modelLegData.size()));


        GLint size;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &size);
        std::cout << "GL_MAX_SHADER_STORAGE_BLOCK_SIZE is " << size << " bytes\n";


	}

	unsigned int CreateComputeShader(std::string shaderSourcePath)
	{
        std::string shaderSourcetemp = Utilities::ReadFileBinary("../Utilities/ShaderSource/" + shaderSourcePath);
        const char* shaderSource = shaderSourcetemp.c_str();

        unsigned int shader;
        int compiled;

        // Create the shader object
        shader = glCreateShader(GL_COMPUTE_SHADER);
        if (shader == 0)
            return shader;

        // Load the shader source
        glShaderSource(shader, 1, &shaderSource, NULL);

        // Compile the shader
        glCompileShader(shader);
        // Check the compile status
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled)
        {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);

            if (infoLen > 1)
            {
                char* infoLog = (char*)malloc(sizeof(char) * infoLen);
                glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
                std::cout << "Error compiling shader : " << infoLog << '\n';
                free(infoLog);
            }
            glDeleteShader(shader);
            return 0;
        }
        return shader;
	}

    unsigned int CreateProgram(std::string shaderSourcePath)
    {
        GLuint program = glCreateProgram();

        GLuint shader = Shader::CreateComputeShader(shaderSourcePath);


        glAttachShader(program, shader);
        glLinkProgram(program);

        glDeleteShader(shader);
        
        return program;
    }

    unsigned int CreateSSBO(SSBOIndex si, uint32_t size)
    {
        unsigned int bufferID;

        glGenBuffers(1, &bufferID);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (unsigned int)si, bufferID);

        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(double) * size, nullptr, GL_DYNAMIC_READ);

        return bufferID;
    }

    unsigned int CreateSSBO(const std::vector<double>& x, SSBOIndex si)
    {
        static std::vector<int> bufferIDs = { -1, -1, -1, -1 };
        int bufferID = bufferIDs[(int)si - 4];

        int totalsize = sizeof(double) * x.size();

        if (bufferID == -1)
        {
            glGenBuffers(1, (unsigned int*)&bufferID);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);
            glBufferData(GL_SHADER_STORAGE_BUFFER, totalsize, nullptr, GL_STREAM_COPY);
            bufferIDs[(int)si - 4] = bufferID;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, totalsize, (void*)x.data());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (unsigned int)si, bufferID);

        return bufferID;
    }

    unsigned int CreateSSBO(const std::vector<std::vector<double>>& x, SSBOIndex si)
    {

        unsigned int bufferID;

        int xSize = x.size();
        int xSize2 = x[0].size();
        int totalsize = sizeof(xSize) * 2 + sizeof(double) * xSize * xSize2;

        glGenBuffers(1, &bufferID);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID);
        glBufferData(GL_SHADER_STORAGE_BUFFER, totalsize, nullptr, GL_STREAM_COPY);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(xSize), (void*)&xSize);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(xSize), sizeof(xSize2), (void*)&xSize2);
        for(int i = 0; i < x.size(); i++)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(xSize) * 2 + i * xSize2 * sizeof(double), xSize2 * sizeof(double), (void*)x[i].data());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (unsigned int)si, bufferID);

        return bufferID;
    }

	std::vector<double> DispatchComputeShader(unsigned int programID, SSBOIndex inputSSBOIndex, const std::vector<double>& inputData, uint32_t size)
	{

        unsigned int inputBuffer = CreateSSBO(inputData, (SSBOIndex)((unsigned int)inputSSBOIndex));

        glUseProgram(programID);
        glDispatchCompute(1, 1, 1);
        //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glFinish();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (unsigned int)(inputSSBOIndex) + 4, SSBOOutput[(unsigned int)(inputSSBOIndex) - 4]);

        GLuint* ptr = (GLuint*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(double) * size, GL_MAP_READ_BIT);

        GLuint err;
        do
        {
            err = glGetError();
            if (err != GLFW_NO_ERROR)
                std::cout << "ERROR = " << err << '\n';
        } while (err != GLFW_NO_ERROR);


        std::vector<double> ret(size);
        memcpy(ret.data(), ptr, sizeof(double) * size);

        std::cout << "PRINTING RESULT OF DISPATCH !! \n";
        for (int i = 0; i < ret.size(); i++)
            std::cout << ret[i] << ',';
        std::cout << '\n';


        if (glUnmapBuffer(GL_SHADER_STORAGE_BUFFER) == GL_FALSE)
            throw std::runtime_error("Failed unmap buffer!");
        

        return ret;
	}
}