#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"

int main()
{

	std::vector<std::string> bodyParts =
	{
		"Head",
		"Torso",
		"Hand",
		"Leg",
		"Negative",
	};

	for (auto& bodyPart : bodyParts)
	{

		// Preprocessing and Augmenting Training Data
		{
			std::vector<std::string> paths = Utilities::GetFilesInDirectory("Dataset\\Training\\" + bodyPart);


			std::vector<Image> images = Image::LoadImages(paths);


			for (int i = 0; i < paths.size(); i++)
			{
				for (int j = 0; j < 5; j++)
				{
					std::cout << "Preprocessing and Augmenting " << images[i].GetName() + '_' + std::to_string(j + 1) << "...\n";
					Image temp = images[i];
					//images[i] = Utilities::GaussianBlur(images[i], 0.84089642, 5);
					temp = Utilities::GaussianBlur(temp, 0.84089642 + Random::RandomProbability(), Random::RandomProbability() < 0.5 ? 3 : 5);
					temp = Utilities::Resize(temp, 128, 128);
					temp = Utilities::Rotate(temp, { -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0 });
					temp = Utilities::HorizontalFlip(temp, 0.5);
					temp.Save("Dataset_Preprocessed\\Training\\" + bodyPart + '\\' + bodyPart + temp.GetName() + '_' + std::to_string(j + 1) + ".png");
				}
			}
		}

		// Preprocessing Testing Data
		{
			std::vector<std::string> paths = Utilities::GetFilesInDirectory("Dataset\\Testing\\" + bodyPart);

			std::vector<Image> images = Image::LoadImages(paths);


			for (int i = 0; i < paths.size(); i++)
			{
				std::cout << "Preprocessing " << images[i].GetName() << "...\n";
				Image temp = images[i];
				temp = Utilities::GaussianBlur(temp, 0.84089642, 5);
				temp = Utilities::Resize(temp, 128, 128);
				temp.Save("Dataset_Preprocessed\\Testing\\" + bodyPart + '\\' + bodyPart + temp.GetName() + ".png");
			}
		}
	}


	return 0;
}