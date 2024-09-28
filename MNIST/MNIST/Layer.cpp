#include "Layer.h"

Layer::Layer(double* input, const int inSize, const int outSize)
{
    this->input = input;
    this->inSize = inSize;
    this->outSize = outSize;

    output = new double[outSize];
    wsum = new double[outSize];

    weights = new double[outSize * inSize];
    bias = new double[outSize];

    error = new double[outSize];
    acceleration = new double[outSize * inSize];

    for (int next = 0; next < outSize; next++) {
        for (int prev = 0; prev < inSize; prev++) {
            weights[prev + next * inSize] = (double)(std::rand() % 2000) / 1000 - 1;
            acceleration[prev + next * inSize] = 0;
        }
        bias[next] = (double)(std::rand() % 2000) / 1000 - 1;
    }
}

//Layer::~Layer()
//{
//    delete[] output;
//    delete[] wsum;
//
//    delete[] weights;
//    delete[] bias;
//
//    delete[] error;
//    delete[] acceleration;
//}

void Layer::calculate()
{
    //������ ����
    for (int next = 0; next < outSize; next++) {
        // ��������� ���������� ���������� �����
        wsum[next] = bias[next];
        // ���������� �� ������� ������� �������� � ��������� ��������
        for (int prev = 0; prev < inSize; prev++) 
            wsum[next] += input[prev] * weights[prev + next * inSize];

        // ��������� ������� ��������� (Sigmoid)
        output[next] = function(wsum[next]);
    }
}

void Layer::recalculateError(double* nextWeights, double* nextError, int nextSize)
{
    for (int curr = 0; curr < outSize; curr++) {
        // ������������� ������ � 0
        error[curr] = 0;
        // ��������� �������� ������ �� ���������� ���� ��� ��������� ������ �������� ����
        for (int next = 0; next < nextSize; next++)
            error[curr] += nextError[next] * nextWeights[curr + next * outSize];
    }
}

void Layer::recalculateLastLayerError(double* rightResults)
{
    // ����������� ������� ��� ������ ���������� ����
    for (int curr = 0; curr < outSize; curr++)
        error[curr] = (rightResults[curr] - output[curr]);
    
}

void Layer::recalculateWeights(double momentum, double learningSpeed)
{
    for (int next = 0; next < outSize; next++) {
        //����� �������� ����� ������ ��� �� �������������
        double delta = error[next] * learningSpeed * derivative(wsum[next]);

        //���� ���������� �����
        for (int prev = 0; prev < inSize; prev++) {
            //���������� ������� ����
            acceleration[prev + next * inSize] *= momentum;
            //���������� ��������� ���� ��� ������� �������� � ���������
            acceleration[prev + next * inSize] += (1 - momentum) * input[prev] * delta;
            //��������� ���� ����������
            weights[prev + next * inSize] += acceleration[prev + next * inSize];
        }
        //���������� ���������� (bias)
        bias[next] += delta;
    }
}


// �������
double* Layer::getOutput()
{
    return output;
}

double* Layer::getError()
{
    return error;
}

double* Layer::getWeights()
{
    return weights;
}

int Layer::getInSize()
{
    return inSize;
}

int Layer::getOutSize()
{
    return outSize;
}


// ������� ���������
double Layer::function(double x) {
    return 1.0 / (1 + std::exp(-x));
}

// ����������� ������� ���������
double Layer::derivative(double x) {
    double exp = std::exp(-x);
    return exp / std::pow(1 + exp, 2);
}
