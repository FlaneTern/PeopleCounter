# PeopleCounter
Source code untuk final project Mata Kuliah Penglihatan Komputer dan Analisis Citra (PKAC).

Dibuat dalam bahasa C++, versi C++20, menggunakan IDE Microsoft Visual Studio 2022.

Project yang menghasilkan static library adalah :
- HistogramOfOrientedGradients
- RBFKernelSVM
- SlidingWindow
- Utilities

Project yang menghasilkan executable adalah :
- DataPreprocessing
- Demo
- ModelTesting
- ModelTraining

Langkah-langkah untuk menjalankan source code :
- Buka file .sln menggunakan Microsft Visual Studio 2022. 
- Build Solution
- Pilih salah satu dari keempat project yang menghasilkan executable sebagai startup project
- Jalankan Start Debugging


Seluruh file selain source code, seperti file model dan citra, diletakkan pada folder WorkingDirectory. Folder ini akan menjadi relative path dari program. 


Dependency :
- OpenCV 4.7.0
- OpenGL (jika menggunakan compute shader)