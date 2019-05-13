#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <map> 

#include <vl/svm.h>
#include "ReadHelper.cpp"
#include "TrainHelper.cpp"
#include "ClassifyHelper.cpp"

using namespace svm;

void Train(const string &image_train_path, const string &norm_param_path, const string &train_model_path)
{
	cout << "Training started..." << endl;

	Data data = read_mnist(image_train_path); //Читаем mnist файл
	Matrix x = get<0>(data); //яркости пикселей
	Array y = get<1>(data); //метки
	double lambda = 0.01;
	vl_size const numData = get<2>(data); //60000 обучающих изображений
	vl_size const dimension = 784; //28*28 пикселей

	auto out = Normalize(x); //нормализуем обучающую выборку
	x = get<0>(out);
	double mean = get<1>(out);
	double std_dev = get<2>(out);
	SaveNormParams(norm_param_path, mean, std_dev);

	tuple<Matrix, Array> trainResult = trainModel(x, y, lambda); //начинаем обучение

	SaveModel(trainResult, train_model_path); //сохраняем модель
}

void Classify(const string &image_class_path, const string &model_path, const string &norm_param_path)
{
	cout << "Classify started..." << endl;

	Data data = read_mnist(image_class_path); //Читаем mnist файл
	//Array y = get<1>(data); 
	vl_size const numData = get<2>(data); //10000 изображений 
	vl_size const dimension = 784; // 28*28 пикселей

	auto out = LoadNormParams(norm_param_path); // считываем параметры нормализации
	double mean = get<0>(out);
	double std_dev = get<1>(out);

	Model models = read_model(model_path, dimension); //считываем натренерованную модель
	Matrix x = get<0>(data);

	x = Normalize(x, mean, std_dev); //нормализуем значения входных значений
	Array UniqueSortedY = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //получам список уникальных значений labels

	vector<int> labels = recognize(models, x, UniqueSortedY); //распознаем

	string result_path = "D:\\mnist\\result.txt";
	SaveResults(labels, result_path);

	cout << "Done." << endl;
}

int main(int argc, char* argv[])
{
	string mode = argv[1];

	if (mode == "train"){
		//path to image, path to save Norm, path to trainModel
		Train(argv[2], argv[3], argv[4]);
	}
	else if (mode == "class"){
		//path to image,  path to trainModel, path to pathNorm
		Classify(argv[2], argv[3], argv[4]);
	}
	else
		cout << "Please, enter true values" << endl;

	system("pause");
	return 0;
}