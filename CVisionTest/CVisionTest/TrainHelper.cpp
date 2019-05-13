#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vl/svm.h>
#include "ReadHelper.h"

using namespace cv;
using namespace svm;

namespace {
	inline Array GetUniqueLabels(Array &y) {
		Array labels(y);

		sort(labels.begin(), labels.end());
		auto it = unique(labels.begin(), labels.end());
		labels.resize(distance(labels.begin(), it));
		return labels; //получаем на выходе массив уникальных значений из всех предложенных
	}

	inline tuple<Array, double> train(Matrix x, Array &y, double &lambda){
		double bias;
		Array model;
		Array UniqueSortedY = GetUniqueLabels(y), biases;
		int dimension = x[1].size(), numData = x.size();
		Array subX(dimension * numData), subY(numData);

		for (vl_size i = 0; i < numData; ++i) {
			for (vl_size j = 0; j < dimension; ++j) {
				subX[i * dimension + j] = x[i][j]; //трансформация в одномерный массив
			}
			subY[i] = y[i];		
		}

		VlSvm * svm = vl_svm_new(VlSvmSolverSgd, subX.data(), dimension, numData, subY.data(), lambda); //обучение
		vl_svm_train(svm);

		bias = vl_svm_get_bias(svm);

		double const * raw_model = vl_svm_get_model(svm);
		for (vl_size i = 0; i < dimension; ++i) {
			model.push_back(raw_model[i]);
		}

		return make_tuple(model, bias);
	}

	inline tuple<Matrix, Array> trainModel(Matrix &x, Array &y, double &lambda)
	{
		Array UniqueSortedY = GetUniqueLabels(y), biases;
		Matrix models;

		cout << "Study begin..." << endl;
		for (size_t i = 0; i < UniqueSortedY.size(); ++i) {
			for (size_t j = i + 1; j < UniqueSortedY.size(); ++j) {

				Matrix subX;
				Array subY;
				for (size_t k = 0; k < x.size(); ++k) { //выборка по канкретным меткам(к примеру все 0)
					if (y[k] == UniqueSortedY[i]) {
						subX.push_back(x[k]);
						subY.push_back(-1);
					}

					if (y[k] == UniqueSortedY[j]) {
						subX.push_back(x[k]);
						subY.push_back(1);
					}
				}
				tuple<Array, double> svm = train(subX, subY, lambda);
				models.push_back(get<0>(svm));
				biases.push_back(get<1>(svm)); 
			}
		}
		cout << "Study end." << endl;

		return make_tuple(models, biases);
	}

	Matrix Normalize(const Matrix &x, double mean, double std_dev) {
		Matrix result(x);

		cout << "Normalize begin..." << endl;
		for (size_t i = 0; i < result.size(); ++i) {
			for (size_t j = 0; j < result.at(0).size(); ++j) {
				result[i][j] = (result[i][j] - mean) / std_dev;
			}
		}
		cout << "Normalize end." << endl;

		return result;
	}

	tuple<Matrix, double, double> Normalize(const Matrix &x) { //нормализация парамеров
		double sum = 0;
		double sqrSum = 0;
		double bnn = 0;

		for (auto &row : x) {
			for (auto value : row) {
				sum += value;
				sqrSum += value * value;
				++bnn;
			}
		}

		double mean = sum / bnn;
		double variance = (sqrSum / bnn) - mean * mean;
		double std_dev = sqrt(variance);

		Matrix result = Normalize(x, mean, std_dev);

		return make_tuple(result, mean, std_dev);
	}

	void SaveNormParams(const string &path, double mean, double std_dev) { 
		ofstream output(path);
		output << mean << " " << std_dev << endl;
		cout << "Normalize saved." << endl;
	}

	tuple<double, double> LoadNormParams(const string &path) {
		double mean, std_dev;
		ifstream input(path);

		if (!(input >> mean >> std_dev)) {
			throw length_error("incorrect normalization file " + path);
		}
		cout << "Norm. params loaded." << endl;

		return make_tuple(mean, std_dev);
	}


	void SaveModel(tuple<Matrix, Array> model, const string &train_model_path)
	{
		cout << "saving model... " << endl;

		Matrix SVMmodels = get<0>(model);
		Array biases = get<1>(model);
		ofstream output(train_model_path, ios::binary);

		output << biases.size() << endl;

		for (int i = 0; i < biases.size(); i++) {
			output << biases[i] << endl;
		}
		cout << "Biases saved." << endl;

		for (int i = 0; i < biases.size(); i++) {
			for (int j = 0; j < SVMmodels[1].size(); j++) {
				if (j == SVMmodels[1].size() - 1)
					output << SVMmodels[i][j];
				else
					output << SVMmodels[i][j] << endl; //сохраняем значения весов модели построчно
			}
			output << endl;
		}
		cout << "Models saved." << endl;
	}
}