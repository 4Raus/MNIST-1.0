#pragma once
#include <vector>
#include <iostream>
#include <iomanip>

#include "Layer.h"

struct DataSetRow
{
	double* input;
	double* output;
	DataSetRow(double* input, double* output) {
		this->input = input;
		this->output = output;
	}
};

class Network
{
private:
	double* input = nullptr;
	int inSize;

	std::vector<Layer> layers;

public:
	
	void addLayer(int size);

	void setInput(int i, double value);
	double getOutput(int i);

	void calculate();
	void learn(std::vector<DataSetRow> dataset, double exitError, double momentum, double learningSpeed);
};

