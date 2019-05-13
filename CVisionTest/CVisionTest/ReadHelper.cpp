#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vl/svm.h>
#include "ReadHelper.h"


using namespace cv;
using namespace svm;

namespace {

	inline Data read_mnist(const  string &data_path)
	{
		string help_string;
		int count = 0;
		vector<string> paths;
		Array labels;

		ifstream file(data_path, ios::binary);
		if (file.is_open())
		{
			while (getline(file, help_string))
			{
				if (count % 2 == 0)	{	//записываем ссылки
					stringstream s(help_string);
					string trimmed_string;
					s >> trimmed_string;
					paths.push_back(trimmed_string);
				}

				else					//записываем метки
					labels.push_back(atof(help_string.c_str()));
				count++;
			}

			int numData = count / 2; //кол-во изображений
			Matrix x; //двумерный массив с изображениями

			for (int i = 0; i < numData; i++){
				if ((i + 1) % 10000 == 0) { // Выводим прогресс загрузки
					cout << "loaded images " << i + 1 << endl;
				}

				Mat imageBefor, imageAfter;
				imageBefor = imread(paths[i], IMREAD_GRAYSCALE); //в серых тонах

				if (imageBefor.empty())
					throw runtime_error("Unable to open file `" + paths[i] + "`!");
				else
				{
					if (imageBefor.rows != 28 && imageBefor.cols != 28)  //если изображене НЕ 28*28 пикселей, то сжимаем до этого размера
						resize(imageBefor, imageAfter, Size(28, 28));
					else
						imageAfter = imageBefor;

					Array xImage;
					for (int i = 0; i < imageAfter.rows; ++i) {
						for (int j = 0; j < imageAfter.cols; ++j) {
							xImage.push_back((double)imageAfter.at<uchar>(i, j)); //конвертируем в double
						}
					}

					x.push_back(xImage);
				}
			}
			cout << "All images loaded" << endl;
			return make_tuple(x, labels, numData);
		}
		else {
			throw runtime_error("Unable to open file `" + data_path + "`!");
		}
	}

	inline Model read_model(const string &model_path, vl_size const dimension)
	{
		Matrix model;
		Array biases;

		ifstream file(model_path, ios::binary);
		if (file.is_open())
		{
			string number_of_biases, help_string;
			int number_of_models;
			int counter = 0;

			getline(file, number_of_biases); //узнаем кол-во bias

			biases.resize(stoi(number_of_biases)); 

			for (int i = 0; i < stoi(number_of_biases); i++) //читаем bias
			{
				getline(file, help_string);
				biases[i] = stod(help_string);
			}

			model.resize(stoi(number_of_biases), Array(dimension));
			for (int i = 0; i < stoi(number_of_biases); i++) //читаем model
			{
				for (int j = 0; j < dimension; j++) 
				{
					getline(file, help_string);
					model[i][j] = stod(help_string); //заполняем модель
				}
			}

			return make_tuple(model, biases);
		}
		else {
			throw runtime_error("Unable to open file `" + model_path + "`!");
		}
	}
}