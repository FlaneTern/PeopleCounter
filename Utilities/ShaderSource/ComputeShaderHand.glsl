#version 460 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 2) readonly buffer Input1
{
    int LagrangeMultiplierCount;
    int FeatureCount;
    double elements[];
} input1;

layout(std430, binding = 6) readonly buffer Input2
{
    double elements[];
} input2;

layout(std430, binding = 10) writeonly buffer OutputData
{
    double data[];
} output_data;

void main()
{
    double acc = 0;
    double temp = 0;
    for(int i = 0; i < input1.LagrangeMultiplierCount; i++)
    {
        acc = 0;
        temp = 0;
        
        for(int j = 0; j < input1.FeatureCount; j++)
        {
            temp = input1.elements[i * input1.FeatureCount + j] - input2.elements[j];
            acc += temp * temp;
        }
        output_data.data[i] = acc;
    }
}