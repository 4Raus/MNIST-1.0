#include <string>
#include <iostream>
#include <fstream>

#include "Network.h"


int main()
{
	const int imgSize = 28;
	const int sqr = imgSize * imgSize;

	// Создаем нейросеть
	std::cout << "Network building..." << std::endl;
	Network network;
	network.addLayer(sqr);
	network.addLayer(sqr / 4);
	network.addLayer(sqr / 16);
	network.addLayer(10);

	std::cout << "Dataset loading..." << std::endl;

	std::ifstream fin;
	std::vector<DataSetRow> dataset;
	double num;

	for (int i = 0; i < 1000; i++) {
		fin.open(("dataset/" + std::to_string(i) + ".txt"));
		if (!fin.is_open()) {
			std::cout << "Loading error!" << std::endl;
			break;
		}

		double* in = new double[sqr];
		for (int j = 0; j < sqr; j++) {
			fin >> num;
			in[j] = num;
		}
		double* out = new double[10];
		for (int j = 0; j < 10; j++) {
			fin >> num;
			out[j] = num;
		}
		fin.close();

		dataset.push_back(DataSetRow(in, out));
	}
	std::cout << "Start learning..." << std::endl;

	network.learn(dataset, 2e-2, 0.3, 0.1);

	for (int i = 0; i < 10; i++) {
		fin.open(("dataset/" + std::to_string(i) + ".txt"));
		if (!fin.is_open()) {
			std::cout << "Loading error!" << std::endl;
			break;
		}

		for (int j = 0; j < sqr; j++) {
			fin >> num;
			network.setInput(j, num);
		}
		network.calculate();

		std::cout << "============================" << i << "============================" << std::endl;
		for (int j = 0; j < 10; j++) {
			fin >> num;
			std::cout << num << " " << network.getOutput(j) << std::endl;
		}
		fin.close();
	}

}