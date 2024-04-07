#include "pch.h"
#include "Utilities.h"
#include "HistogramOfOrientedGradients.h"

int main()
{

#if 0
	std::vector<std::string> paths = Utilities::GetFilesInDirectory("faceimages\\data\\test\\human_body_parts face\\");


	std::vector<Image> images = Image::LoadImages(paths);


	for (int i = 0; i < paths.size(); i++)
	{
		images[i] = Utilities::Resize(images[i], 128, 128);
		images[i] = Utilities::GaussianBlur(images[i], 0.84089642, 5);
		images[i] = Utilities::Rotate(images[i], { 20.0 });
		images[i] = Utilities::HorizontalFlip(images[i], 1);
		images[i].Save(paths[i] + "modified.png");
	}
#elif 1
	Image image = Image::LoadImage("butterfly.jpg");
	//image = Utilities::Resize(image, 128, 128);
	image = Utilities::Resize(image, 241, 200);
	//image = Utilities::Crop(image, 100, 100, 300, 300);
	//image = Utilities::GaussianBlur(image, 0.84089642, 5);
	//image = Utilities::Rotate(image, { 20.0 });
	//image = Utilities::HorizontalFlip(image, 1);

	//SingleHOGExecutor hoge(image);
	//hoge.Execute();
	image.Draw();
	Utilities::DrawFlush();

	SlidingHOGExecutor hoges(image, 1);
	hoges.Execute();

	//hoges.GetHOGs();


#endif
	


	return 0;
}