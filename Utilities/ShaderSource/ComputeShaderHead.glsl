#version 460 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 1) readonly buffer Input1
{
    int LagrangeMultiplierCount;
    int FeatureCount;
    double elements[];
} input1;

layout(std430, binding = 5) readonly buffer Input2
{
    double elements[];
} input2;

layout(std430, binding = 9) writeonly buffer OutputData
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
            output_data.data[6] = j;
            output_data.data[7] = i * input1.FeatureCount + j;
        }
        output_data.data[i] = acc;
        output_data.data[0] = input1.LagrangeMultiplierCount;
        output_data.data[1] = input1.FeatureCount;
        output_data.data[2] = input1.elements.length();
        output_data.data[3] = input2.elements.length();
        output_data.data[4] = output_data.data.length();
        output_data.data[5] = i;

    }

    for(int i = 0; i < input1.LagrangeMultiplierCount; i++)
    {
        output_data.data[i] = input1.LagrangeMultiplierCount;
    }
}