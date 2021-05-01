/**
 * @file neuronet.cpp
 * @author Jacob Borden (amenra.beats@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "pch.h"
#include "neuronet.h"

NeuroNet::NeuroNetLayer::NeuroNetLayer()
{
}

NeuroNet::NeuroNetLayer::~NeuroNetLayer()
{
}

void NeuroNet::NeuroNetLayer::ResizeLayer(int pInputSize, int pLayerSize)
{
	this->LayerSize = pLayerSize;
	this->InputSize = pInputSize;
	this->InputMatrix.Resize(1, this->InputSize);
	this->WeightMatrix.Resize(this->InputSize, this->LayerSize);
	this->BiasMatrix.Resize(1, this->LayerSize);
	this->OutputMatrix.Resize(1, this->LayerSize);
	this->Weights.WeightCount = this->LayerSize * this->InputSize;
	this->Biases.BiasCount = this->LayerSize;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::CalculateOutput()
{
	this->OutputMatrix = this->InputMatrix * this->WeightMatrix + this->BiasMatrix;
	return this->OutputMatrix;
}

bool NeuroNet::NeuroNetLayer::SetInput(Matrix::Matrix<float> pInputMatrix)
{
	if (pInputMatrix.Rows() != 1 || pInputMatrix.Columns() != this->InputMatrix.Columns())
		return false;
	this->InputMatrix = pInputMatrix;
	return true;
}

int NeuroNet::NeuroNetLayer::WeightCount()
{
	return this->Weights.WeightCount;
}

int NeuroNet::NeuroNetLayer::BiasCount()
{
	return this->Biases.BiasCount;
}

bool NeuroNet::NeuroNetLayer::SetWeights(LayerWeights pWeights)
{
	if (pWeights.WeightCount != this->Weights.WeightCount)
		return false;
	this->Weights = pWeights;
	int k = 0;
	for (int i = 0; i < this->WeightMatrix.Rows(); i++)
		for (int j = 0; j < this->WeightMatrix.Columns(); j++)
		{
			this->WeightMatrix[i][j] = this->Weights.WeightsVector[k];
			k++;
		}
	return true;
}

bool NeuroNet::NeuroNetLayer::SetBiases(LayerBiases pBiases)
{
	if (pBiases.BiasCount != this->Biases.BiasCount)
		return false;
	this->Biases = pBiases;
	int k = 0;
	for (int i = 0; i < this->BiasMatrix.Rows(); i++)
		for (int j = 0; j < this->BiasMatrix.Columns(); j++)
		{
			this->BiasMatrix[i][j] = this->Biases.BiasVector[k];
			k++;
		}
	return true;
}

NeuroNet::NeuroNet::NeuroNet()
{
}

NeuroNet::NeuroNet::NeuroNet(int pLayerCount)
{
}

NeuroNet::NeuroNet::~NeuroNet()
{
}

bool NeuroNet::NeuroNet::ResizeLayer(int pLayerIndex, int pLayerSize)
{
	return false;
}

void NeuroNet::NeuroNet::SetInputSize(int pInputSize)
{
}

void NeuroNet::NeuroNet::ResizeNeuroNet(int pLayerCount)
{
}

bool NeuroNet::NeuroNet::SetInput(int pInputSize)
{
	return false;
}

Matrix::Matrix<float> NeuroNet::NeuroNet::GetOutput()
{
	return Matrix::Matrix<float>();
}
