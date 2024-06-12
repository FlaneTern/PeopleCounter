#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"

#include "SlidingWindow.h"

static constexpr double s_FarThreshold = 2.0;


struct Assemble
{
	int Index1;
	int Index2;
	double Distance;
};

struct DisjointSet
{
	int Index;
	int Root;
};

static std::vector<Assemble> ComputeBodyPartDistances(const std::vector<BoundingBox>& bodyPart1, const std::vector<BoundingBox>& bodyPart2)
{
	std::vector<Assemble> assembler;
	assembler.reserve(bodyPart1.size() * bodyPart2.size());

	for (int i = 0; i < bodyPart1.size(); i++)
	{
		//double closestDistance = std::numeric_limits<double>::max();

		double xCenterHead = bodyPart1[i].XAnchor + bodyPart1[i].XSize - 1;
		double yCenterHead = bodyPart1[i].YAnchor + bodyPart1[i].YSize - 1;

		for (int j = 0; j < bodyPart2.size(); j++)
		{
			double xCenterTorso = bodyPart2[j].XAnchor + bodyPart2[j].XSize - 1;
			double yCenterTorso = bodyPart2[j].YAnchor + bodyPart2[j].YSize - 1;

			double currentDistance = (xCenterHead - xCenterTorso) * (xCenterHead - xCenterTorso) + (yCenterHead - yCenterTorso) * (yCenterHead - yCenterTorso);

			assembler.push_back({ i, j, currentDistance });
		}
	}

	for (int i = 0; i < assembler.size(); i++)
		std::sort(assembler.begin(), assembler.end(), [](const Assemble& a, const Assemble& b) {return a.Distance > b.Distance; });

	return assembler;
}

// 1 must be torso coz scale threshold
static std::vector<Assemble> AttachBodyParts(const std::vector<BoundingBox>& bodyPart1, const std::vector<BoundingBox>& bodyPart2, int maxAttachedTo1, int maxAttachedTo2)
{


	std::vector<Assemble> assembler = ComputeBodyPartDistances(bodyPart1, bodyPart2);

	std::vector<int> bodyPart1Counter(bodyPart1.size(), maxAttachedTo1);
	std::vector<int> bodyPart2Counter(bodyPart2.size(), maxAttachedTo2);

	for (int i = 0; i < assembler.size(); i++)
	{
		if (assembler[i].Distance > s_FarThreshold * bodyPart1[assembler[i].Index1].XSize)
		{
			assembler.erase(assembler.begin() + i);
			i--;
		}
	}


	for (int i = 0; i < assembler.size(); i++)
	{
		int bodyPart1Index = assembler[i].Index1;
		int bodyPart2Index = assembler[i].Index2;

		bodyPart1Counter[bodyPart1Index]--;
		bodyPart2Counter[bodyPart2Index]--;

		for (int j = i + 1; j < assembler.size(); j++)
		{
			if ((assembler[j].Index1 == bodyPart1Index && bodyPart1Counter[bodyPart1Index] <= 0) ||
				(assembler[j].Index2 == bodyPart2Index && bodyPart2Counter[bodyPart2Index] <= 0))
			{
				assembler.erase(assembler.begin() + j);
				j--;
				continue;
			}

			if (assembler[j].Index1 == bodyPart1Index)
				bodyPart1Counter[bodyPart1Index]--;

			if (assembler[j].Index2 == bodyPart2Index)
				bodyPart2Counter[bodyPart2Index]--;
		}
	}

	return assembler;
}

static std::vector<Assemble> AttachBodyParts(const std::vector<BoundingBox>& bodyPart1, const std::vector<BoundingBox>& bodyPart2, int maxAttachedTo1, int maxAttachedTo2, std::vector<bool> isAttached1, std::vector<bool> isAttached2)
{


	std::vector<Assemble> assembler = ComputeBodyPartDistances(bodyPart1, bodyPart2);

	std::vector<int> bodyPart1Counter(bodyPart1.size(), maxAttachedTo1);
	std::vector<int> bodyPart2Counter(bodyPart2.size(), maxAttachedTo2);

	for (int i = 0; i < assembler.size(); i++)
	{
		if (assembler[i].Distance > s_FarThreshold * bodyPart1[assembler[i].Index1].XSize || 
			isAttached1[assembler[i].Index1] || isAttached1[assembler[i].Index2])
		{
			assembler.erase(assembler.begin() + i);
			i--;
		}
	}


	for (int i = 0; i < assembler.size(); i++)
	{
		int bodyPart1Index = assembler[i].Index1;
		int bodyPart2Index = assembler[i].Index2;

		bodyPart1Counter[bodyPart1Index]--;
		bodyPart2Counter[bodyPart2Index]--;

		for (int j = i + 1; j < assembler.size(); j++)
		{
			if ((assembler[j].Index1 == bodyPart1Index && bodyPart1Counter[bodyPart1Index] <= 0) ||
				(assembler[j].Index2 == bodyPart2Index && bodyPart2Counter[bodyPart2Index] <= 0))
			{
				assembler.erase(assembler.begin() + j);
				j--;
				continue;
			}

			if (assembler[j].Index1 == bodyPart1Index)
				bodyPart1Counter[bodyPart1Index]--;

			if (assembler[j].Index2 == bodyPart2Index)
				bodyPart2Counter[bodyPart2Index]--;
		}
	}

	return assembler;
}

static std::vector<Assemble> AttachBodyPartsSame(const std::vector<BoundingBox>& bodyPart, int maxAttachedTo, std::vector<bool> isAttached)
{
	static constexpr double s_FarThreshold = 2.0;

	std::vector<Assemble> assembler = ComputeBodyPartDistances(bodyPart, bodyPart);

	std::vector<int> bodyPartCounter(bodyPart.size(), maxAttachedTo);

	for (int i = 0; i < assembler.size(); i++)
	{
		if (assembler[i].Distance > s_FarThreshold * bodyPart[assembler[i].Index1].XSize ||
			isAttached[assembler[i].Index1] || isAttached[assembler[i].Index2] ||
			assembler[i].Index1 == assembler[i].Index2)
		{
			assembler.erase(assembler.begin() + i);
			i--;
		}
	}


	for (int i = 0; i < assembler.size(); i++)
	{
		int bodyPart1Index = assembler[i].Index1;
		int bodyPart2Index = assembler[i].Index2;

		bodyPartCounter[bodyPart1Index]--;
		bodyPartCounter[bodyPart2Index]--;

		for (int j = i + 1; j < assembler.size(); j++)
		{
			if ((assembler[j].Index1 == bodyPart1Index && bodyPartCounter[bodyPart1Index] <= 0) ||
				(assembler[j].Index2 == bodyPart2Index && bodyPartCounter[bodyPart2Index] <= 0))
			{
				assembler.erase(assembler.begin() + j);
				j--;
				continue;
			}

			if (assembler[j].Index1 == bodyPart1Index)
				bodyPartCounter[bodyPart1Index]--;

			if (assembler[j].Index2 == bodyPart2Index)
				bodyPartCounter[bodyPart2Index]--;
		}
	}	

	return assembler;
}

static double ComputeIoU(BoundingBox bb1, BoundingBox bb2)
{
	double xLeft = std::max(bb1.XAnchor, bb2.XAnchor);
	double xRight = std::min(bb1.XAnchor + bb1.XSize - 1, bb2.XAnchor + bb2.XSize - 1);
	double yTop = std::max(bb1.YAnchor, bb2.YAnchor);
	double yBottom = std::min(bb1.YAnchor + bb1.YSize - 1, bb2.YAnchor + bb2.YSize - 1);

	if (xRight < xLeft || yBottom < yTop)
		return 0;

	double intersection = (xRight - xLeft) * (yBottom - yTop);
	
	return intersection / (bb1.XSize * bb1.YSize + bb2.XSize * bb2.YSize - intersection);
}

#define HELPER(BODYPART, ITEM) m_##BODYPART##ITEM
#define HELPER2(BODYPART, ITEM) BODYPART##ITEM
#define HELPER3(BODYPART) BODYPART##Offset
#define HELPER4(ASSEMBLE) ASSEMBLE

SlidingWindow::SlidingWindow(
	Image image,
	HOGParameter::Parameters hogParams,
	uint32_t stride,
	double scale, uint32_t gamma, double IoUThreshold,
	HeadClassifier headClassifier,
	HandClassifier handClassifier,
	TorsoClassifier torsoClassifier,
	LegClassifier legClassifier
) : m_Image(image), m_DrawImage(image),
	m_Scale(scale), m_Gamma(gamma), m_IoUThreshold(IoUThreshold),
	m_SHE(image, hogParams, stride),
	m_HeadClassifier(headClassifier),
	m_HandClassifier(handClassifier),
	m_TorsoClassifier(torsoClassifier),
	m_LegClassifier(legClassifier)
{}

void SlidingWindow::GenerateBoundingBoxes()
{
	HOGParameter::Parameters hogParams = m_SHE.GetParams();
	Shape originalImageShape = m_Image.GetShape();

	std::cout << "Prediction starting!\n";

	int currentResizeIteration = 1;
	while (m_SHE.GetShape().x >= hogParams.SlidingWindowSize && m_SHE.GetShape().y >= hogParams.SlidingWindowSize && currentResizeIteration < 11)
	{
		currentResizeIteration++;
		m_SHE.ResetHOGGetterIndex();
		m_SHE.Execute();
		uint32_t currentIndex = m_SHE.GetCurrentHOGGetterIndex();

		Shape currentShape = m_SHE.GetShape();

		Shape boundingBoxSize =
		{
			(uint32_t)(hogParams.SlidingWindowSize * originalImageShape.x / (double)currentShape.x),
			(uint32_t)(hogParams.SlidingWindowSize * originalImageShape.y / (double)currentShape.y),
			1
		};

		std::thread headThread;
		std::thread handThread;
		std::thread torsoThread;
		std::thread legThread;

		int currentIteration = 0;
		do
		{
			std::vector<double> hog = m_SHE.GetHOG(currentIndex).m_HOG;
			m_SHE.AdvanceHOGGetterIndex();
			currentIndex = m_SHE.GetCurrentHOGGetterIndex();

			Shape coords = m_SHE.GetCurrentHOGGetterIndexCoords();

			currentIteration++;

#define PREDICTANDGENERATEBOUNDINGBOX(BODYPART)													   \
			double HELPER(BODYPART, Score) = HELPER(BODYPART, Classifier).Predict(hog);			   \
			/*std::cout << "Prediction Score = " << HELPER(BODYPART, Score) << '\n';*/			   \
			if (HELPER(BODYPART, Score) >= 0.5)													   \
			{																					   \
				Shape coords = m_SHE.GetCurrentHOGGetterIndexCoords();							   \
																								   \
				coords.x = (uint32_t)(coords.x * originalImageShape.x / (double)currentShape.x);   \
				coords.y = (uint32_t)(coords.y * originalImageShape.y / (double)currentShape.y);   \
																								   \
				HELPER(BODYPART, BoundingBoxes).push_back(										   \
					{																			   \
						coords.x,																   \
						coords.y,																   \
						boundingBoxSize.x,														   \
						boundingBoxSize.y,														   \
						HELPER(BODYPART, Score)													   \
					});																			   \
			}



			
			headThread = std::thread([&]() {PREDICTANDGENERATEBOUNDINGBOX(Head)});
			torsoThread = std::thread([&]() {PREDICTANDGENERATEBOUNDINGBOX(Torso)});
			handThread = std::thread([&]() {PREDICTANDGENERATEBOUNDINGBOX(Hand)});
			legThread = std::thread([&]() {PREDICTANDGENERATEBOUNDINGBOX(Leg)});

			headThread.join();
			torsoThread.join();
			handThread.join();
			legThread.join();




		} while (currentIndex != 0);



		m_SHE = SlidingHOGExecutor(m_SHE, m_Scale);
	}
	std::cout << "Prediction done!\n";

	//std::cout << "BoundingBox Count Before Merge (NMS) = " << m_TorsoBoundingBoxes.size() + m_HeadBoundingBoxes.size() + m_HandBoundingBoxes.size() + m_LegBoundingBoxes.size() << '\n';
	MergeBoundingBoxes();
	//std::cout << "BoundingBox Count After Merge (NMS) = " << m_TorsoBoundingBoxes.size() + m_HeadBoundingBoxes.size() + m_HandBoundingBoxes.size() + m_LegBoundingBoxes.size() << '\n';
	AssemblePeopleBoundingBoxes();
	//std::cout << "BoundingBox Count (People Count) = " << m_PersonBoundingBoxes.size() << '\n';
	MergeBoundingBoxes();
	//std::cout << "BoundingBox Count (People Count) after merge again = " << m_PersonBoundingBoxes.size() << '\n';
}

void SlidingWindow::MergeBoundingBoxes()
{

	auto mergebb = [&](std::vector<BoundingBox>& input)
	{
		auto& bbsr = input;
		auto bbs = input;
		bbsr.clear();

		while (!bbs.empty())
		{
			BoundingBox best;
			best.Score = -1;

			int bestIndex = -1;

			for (int i = 0; i < bbs.size(); i++)
			{
				if (bbs[i].Score >= best.Score)
				{
					best = bbs[i];
					bestIndex = i;
				}
			}

			bbs.erase(bbs.begin() + bestIndex);
			bbsr.push_back(best);

			for (int i = 0; i < bbs.size(); i++)
			{
				if (ComputeIoU(best, bbs[i]) >= m_IoUThreshold)
				{
					bbs.erase(bbs.begin() + i);
					i--;
				}
			}
		}
	};

	mergebb(m_HeadBoundingBoxes);
	mergebb(m_HandBoundingBoxes);
	mergebb(m_LegBoundingBoxes);
	mergebb(m_TorsoBoundingBoxes);
	mergebb(m_PersonBoundingBoxes);

}

void SlidingWindow::AssemblePeopleBoundingBoxes()
{
	// index 1 = torso, index 2 = head
	std::vector<Assemble> headToTorso = AttachBodyParts(m_TorsoBoundingBoxes, m_HeadBoundingBoxes, 1, 1);
	// index 1 = torso, index 2 = hand
	std::vector<Assemble> handToTorso = AttachBodyParts(m_TorsoBoundingBoxes, m_HandBoundingBoxes, 2, 1);
	// index 1 = torso, index 2 = leg
	std::vector<Assemble> legToTorso = AttachBodyParts(m_TorsoBoundingBoxes, m_LegBoundingBoxes, 2, 1);

	std::vector<bool> isHeadTorsoLess(m_HeadBoundingBoxes.size(), true);
	for (int i = 0; i < headToTorso.size(); i++)
		isHeadTorsoLess[headToTorso[i].Index2] = false;

	std::vector<bool> isHandTorsoLess(m_HandBoundingBoxes.size(), true);
	for (int i = 0; i < handToTorso.size(); i++)
		isHandTorsoLess[handToTorso[i].Index2] = false;

	std::vector<bool> isLegTorsoLess(m_LegBoundingBoxes.size(), true);
	for (int i = 0; i < legToTorso.size(); i++)
		isLegTorsoLess[legToTorso[i].Index2] = false;

	// index 1 = head, index 2 = hand
	std::vector<Assemble> torsoLessHeadToTorsoLessHand = AttachBodyParts(m_HeadBoundingBoxes, m_HandBoundingBoxes, 1, 2,
		isHeadTorsoLess, isHandTorsoLess);

	// index 1 = head, index 2 = leg
	std::vector<Assemble> torsoLessHeadToTorsoLessLeg = AttachBodyParts(m_HeadBoundingBoxes, m_LegBoundingBoxes, 1, 2,
		isHeadTorsoLess, isLegTorsoLess);
	
	// below is danger territory
	// below is danger territory
	// below is danger territory
	// below is danger territory
	// below is danger territory

	std::vector<bool> isHand_TorsoHeadLess(m_HandBoundingBoxes.size(), true);
	for (int i = 0; i < handToTorso.size(); i++)
		isHand_TorsoHeadLess[handToTorso[i].Index2] = false;
	for (int i = 0; i < torsoLessHeadToTorsoLessHand.size(); i++)
		isHand_TorsoHeadLess[torsoLessHeadToTorsoLessHand[i].Index2] = false;

	// index 1 = hand, index 2 = hand
	std::vector<Assemble> torsoHeadLess_Hand_To_TorsoHeadLess_Hand = AttachBodyPartsSame(m_HandBoundingBoxes, 1, isHand_TorsoHeadLess);

	std::vector<bool> isLeg_TorsoHeadLess(m_LegBoundingBoxes.size(), true);
	for (int i = 0; i < legToTorso.size(); i++)
		isLeg_TorsoHeadLess[legToTorso[i].Index2] = false;
	for (int i = 0; i < torsoLessHeadToTorsoLessLeg.size(); i++)
		isLeg_TorsoHeadLess[torsoLessHeadToTorsoLessLeg[i].Index2] = false;

	// index 1 = leg, index 2 = leg
	std::vector<Assemble> torsoHeadLess_Leg_To_TorsoHeadLess_Leg = AttachBodyPartsSame(m_LegBoundingBoxes, 1, isLeg_TorsoHeadLess);

	// assembling
	// body part index offset : Torso -> Head -> Hand -> Leg
	int torsoOffset = 0;
	int headOffset = m_TorsoBoundingBoxes.size();
	int handOffset = m_TorsoBoundingBoxes.size() + m_HeadBoundingBoxes.size();
	int legOffset = m_TorsoBoundingBoxes.size() + m_HeadBoundingBoxes.size() + m_HandBoundingBoxes.size();

#define ADDOFFSET(ASSEMBLE, BODYPART1, BODYPART2)		\
	for (int i = 0; i < HELPER4(ASSEMBLE).size(); i++) 	\
	{													\
		ASSEMBLE[i].Index1 += HELPER3(BODYPART1);		\
		ASSEMBLE[i].Index2 += HELPER3(BODYPART2);		\
	}

	ADDOFFSET(headToTorso, torso, head);
	ADDOFFSET(handToTorso, torso, hand);
	ADDOFFSET(legToTorso, torso, leg);
	ADDOFFSET(torsoLessHeadToTorsoLessHand, head, hand);
	ADDOFFSET(torsoLessHeadToTorsoLessLeg, head, leg);
	ADDOFFSET(torsoHeadLess_Hand_To_TorsoHeadLess_Hand, hand, hand);
	ADDOFFSET(torsoHeadLess_Leg_To_TorsoHeadLess_Leg, leg, leg);


	std::vector<DisjointSet> disjointSet;
	disjointSet.reserve(m_TorsoBoundingBoxes.size() + m_HeadBoundingBoxes.size() + m_HandBoundingBoxes.size() + m_LegBoundingBoxes.size());
	for (int i = 0; i < m_TorsoBoundingBoxes.size() + m_HeadBoundingBoxes.size() + m_HandBoundingBoxes.size() + m_LegBoundingBoxes.size(); i++)
		disjointSet.push_back({ i, i });

#define MERGE(ASSEMBLE, INDEX1OFFSET, INDEX2OFFSET)																	  \
	for (int i = 0; i < HELPER4(ASSEMBLE).size(); i++)																	  \
		disjointSet[HELPER4(ASSEMBLE)[i].Index2].Root = disjointSet[HELPER4(ASSEMBLE)[i].Index1].Root;

	MERGE(headToTorso, torso, head);
	MERGE(handToTorso, torso, hand);
	MERGE(legToTorso, torso, leg);
	MERGE(torsoLessHeadToTorsoLessHand, head, hand);
	MERGE(torsoLessHeadToTorsoLessLeg, head, leg);
	MERGE(torsoHeadLess_Hand_To_TorsoHeadLess_Hand, hand, hand);
	MERGE(torsoHeadLess_Leg_To_TorsoHeadLess_Leg, leg, leg);

	std::sort(disjointSet.begin(), disjointSet.end(), [](const DisjointSet& a, const DisjointSet& b) {return a.Root < b.Root; });

	for (int i = 0; i < disjointSet.size();)
	{
		int j = i;
		while (j < disjointSet.size() && disjointSet[j].Root == disjointSet[i].Root)
			j++;
		j--;


		int xMin = INT_MAX;
		int yMin = INT_MAX;

		int xMax = INT_MIN + 1;
		int yMax = INT_MIN + 1;

		for (int k = i; k <= j; k++)
		{

#define GETMINMAX(BODYPART, BODYPARTSMALL)																		 						 \
			{																					 						 \
				if ((int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].XAnchor < xMin)													 \
					xMin = (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].XAnchor;													 \
				if ((int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].YAnchor < yMin)													 \
					yMin = (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].YAnchor;													 \
				if ((int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].XAnchor + (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].XSize - 1 > xMax)	 \
					xMax = (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].XAnchor + (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].XSize - 1;	 \
				if ((int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].YAnchor + (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].YSize - 1 > yMax)	 \
					yMax = (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].YAnchor + (int)HELPER(BODYPART, BoundingBoxes)[k - HELPER3(BODYPARTSMALL)].YSize - 1;	 \
			}

			if (k < headOffset)
				GETMINMAX(Torso, torso)
			else if (k < handOffset)
				GETMINMAX(Head, head)
			else if (k < legOffset)
				GETMINMAX(Hand, hand)
			else 
				GETMINMAX(Leg, leg)
		}

		m_PersonBoundingBoxes.push_back({ (uint32_t)xMin, (uint32_t)yMin, (uint32_t)(xMax - xMin + 1), (uint32_t)(yMax - yMin + 1), -1 });
		i = j + 1;
	}

}


void SlidingWindow::DrawBoundingBox(BodyPart bodyPart, uint32_t thickness)
{
	std::vector<double> pixels = m_DrawImage.GetPixels();
	Shape shape = m_Image.GetShape();

	static constexpr double HandBoundingBoxColor[] = { 0, 0, 255 };
	static constexpr double HeadBoundingBoxColor[] = { 0, 255, 0 };
	static constexpr double TorsoBoundingBoxColor[] = { 255, 0, 0 };
	static constexpr double LegBoundingBoxColor[] = { 0, 0, 0 };
	static constexpr double PersonBoundingBoxColor[] = { 255, 255, 255 };

	std::vector<BoundingBox> bbs;
	double color[3];
	if (bodyPart == BodyPart::Head)
	{
		bbs = m_HeadBoundingBoxes;
		memcpy(color, HeadBoundingBoxColor, sizeof(color));
	}
	if (bodyPart == BodyPart::Hand)
	{
		bbs = m_HandBoundingBoxes;
		memcpy(color, HandBoundingBoxColor, sizeof(color));
	}
	if (bodyPart == BodyPart::Torso)
	{
		bbs = m_TorsoBoundingBoxes;
		memcpy(color, TorsoBoundingBoxColor, sizeof(color));
	}
	if (bodyPart == BodyPart::Leg)
	{
		bbs = m_LegBoundingBoxes;
		memcpy(color, LegBoundingBoxColor, sizeof(color));
	}
	if (bodyPart == BodyPart::Person)
	{
		bbs = m_PersonBoundingBoxes;
		memcpy(color, PersonBoundingBoxColor, sizeof(color));
	}

	{																																																			  
		for (int i = 0; i < bbs.size(); i++)																																					  
		{																																																		  
			for (int j = bbs[i].YAnchor; j < bbs[i].YAnchor + bbs[i].YSize - 1; j++)																			  
			{																																																	  
				for (int k = bbs[i].XAnchor - (thickness / 2); k <= bbs[i].XAnchor + (thickness / 2); k++)																		  
				{																																																  
					if (j >= 0 && j < shape.y && k >= 0 && k < shape.x)																																			  
					{																																															  
						int coords = Utilities::CoordsToFlat(k, j, 0, shape);																																	  
						memcpy(&pixels[coords], color, sizeof(color));
					}																																															  
				}																																																  
					
				for (int k = bbs[i].XAnchor + bbs[i].XSize - 1 - (thickness / 2); k <= bbs[i].XAnchor + bbs[i].XSize - 1 + (thickness / 2); k++)  
				{																																																  
					if (j >= 0 && j < shape.y && k >= 0 && k < shape.x)																																			  
					{																																															  
						int coords = Utilities::CoordsToFlat(k, j, 0, shape);																																	  
						memcpy(&pixels[coords], color, sizeof(color));
					}																																															  
				}																																																  
			}																																																	  
				
								
								
			for (int j = bbs[i].XAnchor; j < bbs[i].XAnchor + bbs[i].XSize - 1; j++)																			  
			{																																																	  
				for (int k = bbs[i].YAnchor - (thickness / 2); k <= bbs[i].YAnchor + (thickness / 2); k++)																		  
				{																																																  
					if (j >= 0 && j < shape.x && k >= 0 && k < shape.y)																																			  
					{																																															  
						int coords = Utilities::CoordsToFlat(j, k, 0, shape);																																	  
						memcpy(&pixels[coords], color, sizeof(color));
					}																																															  
				}																																																  
										
				for (int k = bbs[i].YAnchor + bbs[i].YSize - 1 - (thickness / 2); k <= bbs[i].YAnchor + bbs[i].YSize - 1 + (thickness / 2); k++)  
				{																																																  
					if (j >= 0 && j < shape.x && k >= 0 && k < shape.y)																																			  
					{																																															  
						int coords = Utilities::CoordsToFlat(j, k, 0, shape);																																	  
						memcpy(&pixels[coords], color, sizeof(color));
					}																																															  
				}																																																  
			}																																																	  
		}																																																		  
	}

	m_DrawImage = Image(pixels, shape, m_Image.GetName());
}

void SlidingWindow::ClearDrawnBoundingBox()
{
	m_DrawImage = m_Image;
}

void SlidingWindow::Draw()
{
	m_DrawImage.Draw();
}

void SlidingWindow::SaveDrawImage(std::string nameAppend, std::string directoryPath)
{
	m_DrawImage.Save(directoryPath + m_DrawImage.GetName() + nameAppend + ".png");

	std::ofstream out(directoryPath + m_DrawImage.GetName() + nameAppend + ".txt");
	out << m_PersonBoundingBoxes.size();

	std::cout << "Saved " << m_DrawImage.GetName() + nameAppend << "\n";
}