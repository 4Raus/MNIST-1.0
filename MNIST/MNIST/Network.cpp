#include "Network.h"

void Network::addLayer(int size)
{
	if (input == nullptr) {
		// ���� ��� �������� ����, �� ������� ������� ����
		input = new double[size];
		inSize = size;
	}
	else {
		// ���� ��� ��� �� ������ ���� ����� ��������, �� ������������ ��� �������� ���� ������ �������� ����
		if (layers.empty()) layers.push_back(Layer(input, inSize, size));
		else {
			// ����� ������������ ������ ����������� ����
			layers.push_back(Layer(layers[layers.size() - 1].getOutput(), layers[layers.size() - 1].getOutSize(), size));
		}
	}
}

void Network::setInput(int i, double val) // ��������� i-��� �������� ��������
{
	input[i] = val;
}

double Network::getOutput(int i) // ��������� i-��� ��������� ��������
{
	return layers[layers.size() - 1].getOutput()[i];
}

void Network::calculate() // ������ ���������
{
	// ��� ������� ���� �������� ������� �������
	for (int i = 0; i < layers.size(); i++) layers[i].calculate();
}

void Network::learn(std::vector<DataSetRow> dataset, double exitError, double momentum, double learningSpeed) // �������� ���������
{
	int iteration = 0;
	double err;
	do { 
		err = 0;
		// ���������� �� ����� ��������
		for (int i = 0; i < dataset.size(); i++) {
			// ��������� �������� � ���������
			for (int j = 0; j < inSize; j++)
				setInput(j, dataset[i].input[j]);

			// ����������� ���������
			calculate();

			// =============�������� ��������������� ������=============

			// ��������� ������ ��� ���������� ����
			layers[layers.size() - 1].recalculateLastLayerError(dataset[i].output);

			// �������� ������ ��� ������ ������ ��������� (����� ��� �������� ��������� �������)
			for (int j = 0; j < layers[layers.size() - 1].getOutSize(); j++)
				err += std::pow(layers[layers.size() - 1].getError()[j], 2);


			// ��������� ������ ��������� ����� � ����� � ������
			for (int j = layers.size() - 2; j >= 0; j--) 
				layers[j].recalculateError(
					layers[j + 1].getWeights(),
					layers[j + 1].getError(),
					layers[j + 1].getOutSize());

			// ����� ���������� �� ���������, �������� ����
			for (int j = 0; j < layers.size(); j++)
				layers[j].recalculateWeights(momentum, learningSpeed);
		}
		//������������ ��������� ������ (����� ��� �������� ��������� �������)
		// ��� ��� ������ ���������� ��� ������ ������������������ ����������� �� �� ���� �������� �� ����� ������ 
		err = std::sqrt(err / dataset.size() / layers[layers.size() - 1].getOutSize());

		// ����� ������ � ������� ��� ���� ����� ��������� ��������
		std::cout << "Iteration: " << std::setw(7) << iteration << "\tError: " << err << std::endl;
		iteration++;
	} while (err > exitError); // ���� � ��� ������ ������ �������� �������� ������ ��������� �� �����
}
