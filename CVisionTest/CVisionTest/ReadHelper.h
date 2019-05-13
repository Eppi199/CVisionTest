#include <string>
#include <vector>

using namespace std;

namespace svm {

	typedef vector<double> Array;
	typedef vector<vector<double>> Matrix;
	typedef tuple<Matrix, Array, int> Data;
	typedef tuple<Matrix, Array> Model;
}