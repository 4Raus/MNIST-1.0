#include "Network.h"

void Network::addLayer(int size)
{
	if (input == nullptr) {
		// Если нет входного слоя, то создать входной слой
		input = new double[size];
		inSize = size;
	}
	else {
		// Если еще нет ни одного слоя кроме входного, то использовать для создания слоя данные входного слоя
		if (layers.empty()) layers.push_back(Layer(input, inSize, size));
		else {
			// Иначе использовать данные предыдужего слоя
			layers.push_back(Layer(layers[layers.size() - 1].getOutput(), layers[layers.size() - 1].getOutSize(), size));
		}
	}
}

void Network::setInput(int i, double val) // Установка i-ого входного значения
{
	input[i] = val;
}

double Network::getOutput(int i) // Получение i-ого выходного значения
{
	return layers[layers.size() - 1].getOutput()[i];
}

void Network::calculate() // Расчет нейросети
{
	// Для каждого слоя вызываем функцию расчета
	for (int i = 0; i < layers.size(); i++) layers[i].calculate();
}

void Network::learn(std::vector<DataSetRow> dataset, double exitError, double momentum, double learningSpeed) // Обучение нейросети
{
	int iteration = 0;
	double err;
	do { 
		err = 0;
		// Проходимся по всему датасету
		for (int i = 0; i < dataset.size(); i++) {
			// Загружаем значения в нейросеть
			for (int j = 0; j < inSize; j++)
				setInput(j, dataset[i].input[j]);

			// Расчитываем нейросеть
			calculate();

			// =============Обратное распространение ошибки=============

			// Вычисляем ошибку для последнего слоя
			layers[layers.size() - 1].recalculateLastLayerError(dataset[i].output);

			// Получаем данные для оценки ошибки нейросети (нужно для проверки выходного условия)
			for (int j = 0; j < layers[layers.size() - 1].getOutSize(); j++)
				err += std::pow(layers[layers.size() - 1].getError()[j], 2);


			// Вычисляем ошибки остальных слоев с конца в начало
			for (int j = layers.size() - 2; j >= 0; j--) 
				layers[j].recalculateError(
					layers[j + 1].getWeights(),
					layers[j + 1].getError(),
					layers[j + 1].getOutSize());

			// Снова прозодимся по нейросети, обновляя веса
			for (int j = 0; j < layers.size(); j++)
				layers[j].recalculateWeights(momentum, learningSpeed);
		}
		//Обрабатываем оценочную ошибку (нужно для проверки выходного условия)
		// Так как оценка происходит при помощи среднеквадратичной погрешности то на этой итерации мы берем корень 
		err = std::sqrt(err / dataset.size() / layers[layers.size() - 1].getOutSize());

		// Вывод данных в консоль для того чтобы оценивать прогресс
		std::cout << "Iteration: " << std::setw(7) << iteration << "\tError: " << err << std::endl;
		iteration++;
	} while (err > exitError); // Пока у нас ошибка больше значения выходной ошибки итерируем по циклу
}
