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
    //Расчет слоя
    for (int next = 0; next < outSize; next++) {
        // Установка отклонения взвешенной суммы
        wsum[next] = bias[next];
        // Проходимся по массиву входных значений и взвешенно сумируем
        for (int prev = 0; prev < inSize; prev++) 
            wsum[next] += input[prev] * weights[prev + next * inSize];

        // Применяем функцию активации (Sigmoid)
        output[next] = function(wsum[next]);
    }
}

void Layer::recalculateError(double* nextWeights, double* nextError, int nextSize)
{
    for (int curr = 0; curr < outSize; curr++) {
        // Устанавливаем ошибку в 0
        error[curr] = 0;
        // Взвешенно сумируем ошибку со следующего слоя для получения ошибки текущего слоя
        for (int next = 0; next < nextSize; next++)
            error[curr] += nextError[next] * nextWeights[curr + next * outSize];
    }
}

void Layer::recalculateLastLayerError(double* rightResults)
{
    // Специальная функция для ошибки последнего слоя
    for (int curr = 0; curr < outSize; curr++)
        error[curr] = (rightResults[curr] - output[curr]);
    
}

void Layer::recalculateWeights(double momentum, double learningSpeed)
{
    for (int next = 0; next < outSize; next++) {
        //Общий параметр чтобы каждый раз не пересчитывать
        double delta = error[next] * learningSpeed * derivative(wsum[next]);

        //Цикл вычисления весов
        for (int prev = 0; prev < inSize; prev++) {
            //Уменьшение момента веса
            acceleration[prev + next * inSize] *= momentum;
            //Добавление изменения веса для текущей итерации к ускорению
            acceleration[prev + next * inSize] += (1 - momentum) * input[prev] * delta;
            //Изменение веса ускорением
            weights[prev + next * inSize] += acceleration[prev + next * inSize];
        }
        //Вычисление отклонения (bias)
        bias[next] += delta;
    }
}


// Геттеры
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


// Функция активации
double Layer::function(double x) {
    return 1.0 / (1 + std::exp(-x));
}

// Производная функции активации
double Layer::derivative(double x) {
    double exp = std::exp(-x);
    return exp / std::pow(1 + exp, 2);
}
