#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"
#include "RBFKernelSVM.h"

#include "SlidingWindow.h"

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

	while (m_SHE.GetShape().x >= hogParams.SlidingWindowSize && m_SHE.GetShape().y >= hogParams.SlidingWindowSize)
	{
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

		do
		{
			std::vector<double> hog = m_SHE.GetHOG(currentIndex).m_HOG;
			m_SHE.AdvanceHOGGetterIndex();
			currentIndex = m_SHE.GetCurrentHOGGetterIndex();


#define PREDICTANDGENERATEBOUNDINGBOX(BODYPART)													   \
			double HELPER(BODYPART, Score) = HELPER(BODYPART, Classifier).Predict(hog);			   \
			/*std::cout << "Prediction Score = " << HELPER(BODYPART, Score) << '\n';*/			   \
			if (HELPER(BODYPART, Score) >= 0)													   \
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

			PREDICTANDGENERATEBOUNDINGBOX(Head);
			PREDICTANDGENERATEBOUNDINGBOX(Hand);
			PREDICTANDGENERATEBOUNDINGBOX(Torso);
			PREDICTANDGENERATEBOUNDINGBOX(Leg);


		} while (currentIndex != 0);

		m_SHE = SlidingHOGExecutor(m_SHE, m_Scale);
	}

	MergeBoundingBoxes();
	std::cout << "BBS = " << m_HandBoundingBoxes.size() << '\n';
}

void SlidingWindow::MergeBoundingBoxes()
{

#define MERGEBOUNDINGBOXES(BODYPART)							 \
	{															 \
		auto bbs = HELPER(BODYPART, BoundingBoxes);				 \
		HELPER(BODYPART, BoundingBoxes).clear();				 \
																 \
		while (!bbs.empty())									 \
		{														 \
			BoundingBox best;									 \
			best.Score = -1;									 \
																 \
			int bestIndex = -1;									 \
																 \
			for (int i = 0; i < bbs.size(); i++)				 \
			{													 \
				if (bbs[i].Score >= best.Score)					 \
				{												 \
					best = bbs[i];								 \
					bestIndex = i;								 \
				}												 \
			}													 \
																 \
			bbs.erase(bbs.begin() + bestIndex);					 \
			HELPER(BODYPART, BoundingBoxes).push_back(best);	 \
				 												 \
			for (int i = 0; i < bbs.size(); i++)				 \
			{													 \
				if (ComputeIoU(best, bbs[i]) >= m_IoUThreshold)	 \
				{												 \
					bbs.erase(bbs.begin() + i);					 \
					i--;										 \
				}												 \
			}													 \
		}														 \
	}

	MERGEBOUNDINGBOXES(Head);
	MERGEBOUNDINGBOXES(Hand);
	MERGEBOUNDINGBOXES(Torso);
	MERGEBOUNDINGBOXES(Leg);

}

void SlidingWindow::DrawBoundingBox(BodyPart bodyPart, uint32_t thickness)
{
	std::vector<double> pixels = m_Image.GetPixels();
	Shape shape = m_Image.GetShape();

	static constexpr double HandBoundingBoxColor[] = { 0, 0, 255 };
	static constexpr double HeadBoundingBoxColor[] = { 0, 255, 0 };
	static constexpr double TorsoBoundingBoxColor[] = { 255, 0, 0 };
	static constexpr double LegBoundingBoxColor[] = { 255, 255, 255 };


#define DRAWBOUNDINGBOXES(BODYPART)																																												  \
	{																																																			  \
		for (int i = 0; i < HELPER(BODYPART, BoundingBoxes).size(); i++)																																					  \
		{																																																		  \
			for (int j = HELPER(BODYPART, BoundingBoxes)[i].YAnchor; j < HELPER(BODYPART, BoundingBoxes)[i].YAnchor + HELPER(BODYPART, BoundingBoxes)[i].YSize - 1; j++)																			  \
			{																																																	  \
				for (int k = HELPER(BODYPART, BoundingBoxes)[i].XAnchor - (thickness / 2); k <= HELPER(BODYPART, BoundingBoxes)[i].XAnchor + (thickness / 2); k++)																		  \
				{																																																  \
					if (j >= 0 && j < shape.y && k >= 0 && k < shape.x)																																			  \
					{																																															  \
						int coords = Utilities::CoordsToFlat(k, j, 0, shape);																																	  \
						memcpy(&pixels[coords], HELPER2(BODYPART, BoundingBoxColor), sizeof(HELPER2(BODYPART, BoundingBoxColor)));																											  \
					}																																															  \
				}																																																  \
																																																				  \
				for (int k = HELPER(BODYPART, BoundingBoxes)[i].XAnchor + HELPER(BODYPART, BoundingBoxes)[i].XSize - 1 - (thickness / 2); k <= HELPER(BODYPART, BoundingBoxes)[i].XAnchor + HELPER(BODYPART, BoundingBoxes)[i].XSize - 1 + (thickness / 2); k++)  \
				{																																																  \
					if (j >= 0 && j < shape.y && k >= 0 && k < shape.x)																																			  \
					{																																															  \
						int coords = Utilities::CoordsToFlat(k, j, 0, shape);																																	  \
						memcpy(&pixels[coords], HELPER2(BODYPART, BoundingBoxColor), sizeof(HELPER2(BODYPART, BoundingBoxColor)));																											  \
					}																																															  \
				}																																																  \
			}																																																	  \
																																																				  \
																																																				  \
																																																				  \
			for (int j = HELPER(BODYPART, BoundingBoxes)[i].XAnchor; j < HELPER(BODYPART, BoundingBoxes)[i].XAnchor + HELPER(BODYPART, BoundingBoxes)[i].XSize - 1; j++)																			  \
			{																																																	  \
				for (int k = HELPER(BODYPART, BoundingBoxes)[i].YAnchor - (thickness / 2); k <= HELPER(BODYPART, BoundingBoxes)[i].YAnchor + (thickness / 2); k++)																		  \
				{																																																  \
					if (j >= 0 && j < shape.x && k >= 0 && k < shape.y)																																			  \
					{																																															  \
						int coords = Utilities::CoordsToFlat(j, k, 0, shape);																																	  \
						memcpy(&pixels[coords], HELPER2(BODYPART, BoundingBoxColor), sizeof(HELPER2(BODYPART, BoundingBoxColor)));																											  \
					}																																															  \
				}																																																  \
																																																				  \
				for (int k = HELPER(BODYPART, BoundingBoxes)[i].YAnchor + HELPER(BODYPART, BoundingBoxes)[i].YSize - 1 - (thickness / 2); k <= HELPER(BODYPART, BoundingBoxes)[i].YAnchor + HELPER(BODYPART, BoundingBoxes)[i].YSize - 1 + (thickness / 2); k++)  \
				{																																																  \
					if (j >= 0 && j < shape.x && k >= 0 && k < shape.y)																																			  \
					{																																															  \
						int coords = Utilities::CoordsToFlat(j, k, 0, shape);																																	  \
						memcpy(&pixels[coords], HELPER2(BODYPART, BoundingBoxColor), sizeof(HELPER2(BODYPART, BoundingBoxColor)));																											  \
					}																																															  \
				}																																																  \
			}																																																	  \
		}																																																		  \
	}

	if(bodyPart == BodyPart::Head) { DRAWBOUNDINGBOXES(Head) }
	if(bodyPart == BodyPart::Hand) { DRAWBOUNDINGBOXES(Hand) }
	if(bodyPart == BodyPart::Torso) { DRAWBOUNDINGBOXES(Torso) }
	if(bodyPart == BodyPart::Leg) { DRAWBOUNDINGBOXES(Leg) }

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