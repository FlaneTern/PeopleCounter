#include "pch.h"
#include "Utilities.h"

uint64_t Random::RandomNumber(uint64_t min, uint64_t max)
{
	uint64_t temp = s_Random.m_RNG();
	if (temp == UINT64_MAX)
		return min;
	return temp * (double)((int64_t)(max + 1) - min) / UINT64_MAX + min;
}

double Random::RandomProbability()
{
	return (double)s_Random.m_RNG() / UINT64_MAX;
}