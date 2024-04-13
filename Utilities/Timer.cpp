#include "pch.h"
#include "Utilities.h"

Timer::Timer()
	: m_Name("") 
{
	Start();
}

Timer::Timer(std::string name)
	: m_Name(name) 
{
	Start();
}

void Timer::Start()
{
	m_StartTime = std::chrono::high_resolution_clock::now();
}

void Timer::End(bool log)
{
	m_TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_StartTime).count();
	if (log)
		std::cout << "Timer for Function " << m_Name << " : " << m_TotalTime << " microseconds.\n";
}