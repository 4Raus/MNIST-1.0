#pragma once

#include <random>
class Layer
{
private:

	int inSize;
	int outSize;

	double* input;
	double* output;
	double* wsum;

	double* weights;
	double* bias;

	double* error;
	double* acceleration;

	double function(double x);
	double derivative(double x);

public:
	Layer(double* input, const int inSize, const int outSize);
	//~Layer();

	void calculate();
	void recalculateError(double* nextWeights, double* nextError, int nextSize);
	void recalculateLastLayerError(double* rightResults);
	void recalculateWeights(double momentum, double learningSpeed);
	double* getOutput();
	double* getError();
	double* getWeights();

	int getInSize();
	int getOutSize();
};

