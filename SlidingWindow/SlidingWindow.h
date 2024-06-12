#pragma once

struct BoundingBox
{
	uint32_t XAnchor;
	uint32_t YAnchor;
	uint32_t XSize;
	uint32_t YSize;

	double Score;
};


typedef RBFKernelSVM HeadClassifier;
typedef RBFKernelSVM HandClassifier;
typedef RBFKernelSVM TorsoClassifier;
typedef RBFKernelSVM LegClassifier;

typedef BoundingBox HeadBoundingBox;
typedef BoundingBox HandBoundingBox;
typedef BoundingBox TorsoBoundingBox;
typedef BoundingBox LegBoundingBox;
typedef BoundingBox PersonBoundingBox;

class SlidingWindow
{
public:

	enum class BodyPart
	{
		Head = 0,
		Hand = 1,
		Torso = 2,
		Leg = 3,
		Person = 4
	};

	SlidingWindow(
		Image image,
		HOGParameter::Parameters hogParams,
		uint32_t stride, 
		double scale, uint32_t gamma, double IoUThreshold,
		HeadClassifier headClassifier,
		HandClassifier handClassifier,
		TorsoClassifier torsoClassifier,
		LegClassifier legClassifier
	);

	void GenerateBoundingBoxes();

	void DrawBoundingBox(BodyPart bodyPart, uint32_t thickness);
	void ClearDrawnBoundingBox();

	void Draw();
	void SaveDrawImage();

private:
	Image m_Image;

	SlidingHOGExecutor m_SHE;

	HeadClassifier m_HeadClassifier;
	HandClassifier m_HandClassifier;
	TorsoClassifier m_TorsoClassifier;
	LegClassifier m_LegClassifier;

	std::vector<HeadBoundingBox> m_HeadBoundingBoxes;
	std::vector<HandBoundingBox> m_HandBoundingBoxes;
	std::vector<TorsoBoundingBox> m_TorsoBoundingBoxes;
	std::vector<LegBoundingBox> m_LegBoundingBoxes;
	std::vector<PersonBoundingBox> m_PersonBoundingBoxes;

	uint32_t m_Gamma;
	double m_IoUThreshold;
	double m_Scale;
	//Image m_ImageScaled;
	//Shape m_ShapeScaled;

	Image m_DrawImage;

	void MergeBoundingBoxes();
	void AssemblePeopleBoundingBoxes();



};