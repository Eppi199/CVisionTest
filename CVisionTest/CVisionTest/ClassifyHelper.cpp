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
#include <unordered_map>

using namespace cv;
using namespace svm;

namespace{
	inline double sum(Array x, Array model)
	{
		double resultSum = 0;
		for (int i = 0; i < x.size(); ++i) {
			resultSum += x[i] * model[i]; //сумматор
        }
		return resultSum;
	}

	vector<int> Predict(Matrix &x, Array &model, double bias) { //бинарная классфикация
		if (x.empty()) {
			return{};
		}

		int nb_dim = x[0].size();
		vector<int> predictions;
		for (int i = 0; i < x.size(); ++i) {
			auto &input = x[i];
			double dot_product = sum(input, model);
			int prediction = (dot_product + bias > 0) ? 1 : -1;
			predictions.push_back(prediction);
		}

		return predictions;
	}

	vector<int> recognize(Model &models, Matrix &x, Array &UniqueSortedY) { //классификация
		if (x.empty()) {
			return{};
		}

		Matrix models_ = get<0>(models);
		Array biases = get<1>(models);
		int counter = 0;
		vector<std::vector<int>> raw_results;

		// при классификации используется большинство голосов
		for (int i = 0; i < UniqueSortedY.size(); ++i) {
			for (int j = i + 1; j < UniqueSortedY.size(); ++j) {
				
				vector<int> sub_predictions = Predict(x, models_[counter], biases[counter]);
				vector<int> raw_prediction;
				for (auto prediction : sub_predictions) {
					raw_prediction.push_back(prediction == -1 ? UniqueSortedY[i] : UniqueSortedY[j]);
				}
				raw_results.push_back(raw_prediction);
				++counter;
			}
		}

		vector<int> resulst;
		for (int i = 0; i < x.size(); ++i) {
			unordered_map<int, int> dict;
			int commonest;
			int maxcount = 0;
			for (auto &raw_result : raw_results) {
				if (++dict[raw_result[i]] > maxcount) {
					commonest = raw_result[i];
					maxcount = dict[raw_result[i]];
				}
			}

			resulst.push_back(commonest);
		}

		return resulst;
	}

	void SaveResults(vector<int> labels, const string &rec_labels_path)
	{
		cout << "saving results... " << endl;

		ofstream output(rec_labels_path, ios::binary);

		for (int i = 0; i < labels.size(); i++) {
			output << labels[i] << endl;
		}
		cout << "labels saved." << endl;

	}
}