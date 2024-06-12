#version 460 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer Input1
{
    int size;
    int temp;
    double elements[];
} input1;

layout(std430, binding = 1) readonly buffer Input2
{
    int size;
    int temp;
    double elements[];
} input2;

layout(std430, binding = 2) writeonly buffer OutputData
{
    double data;
} output_data;

void main()
{
    double acc = 0;
    for(int i = 0; i < input1.size; i++)
    {
        acc += input1.elements[i] * input2.elements[i];
    }
    output_data.data = acc;
}