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
	this->vLayerSize = pLayerSize;
	this->InputSize = pInputSize;
	this->InputMatrix.resize(1, this->InputSize);
	this->WeightMatrix.resize(this->InputSize, this->vLayerSize);
	this->BiasMatrix.resize(1, this->vLayerSize);
	this->OutputMatrix.resize(1, this->vLayerSize);
	this->Weights.WeightCount = this->vLayerSize * this->InputSize;
	this->Biases.BiasCount = this->vLayerSize;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::CalculateOutput()
{
	this->OutputMatrix = this->InputMatrix * this->WeightMatrix + this->BiasMatrix;
	return this->OutputMatrix;
}

Matrix::Matrix<float> NeuroNet::NeuroNetLayer::ReturnOutputMatrix()
{
	return this->OutputMatrix;
}

bool NeuroNet::NeuroNetLayer::SetInput(Matrix::Matrix<float> pInputMatrix)
{
	if (pInputMatrix.rows() != 1 || pInputMatrix.cols() != this->InputMatrix.cols())
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

int NeuroNet::NeuroNetLayer::LayerSize()
{
	return this->vLayerSize;
}

bool NeuroNet::NeuroNetLayer::SetWeights(LayerWeights pWeights)
{
	if (pWeights.WeightCount != this->Weights.WeightCount)
		return false;
	this->Weights = pWeights;
	int k = 0;
	for (int i = 0; i < this->WeightMatrix.rows(); i++)
		for (int j = 0; j < this->WeightMatrix.cols(); j++)
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
	for (int i = 0; i < this->BiasMatrix.rows(); i++)
		for (int j = 0; j < this->BiasMatrix.cols(); j++)
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
	this->NeuroNetVector.resize(pLayerCount);
}

NeuroNet::NeuroNet::~NeuroNet()
{
}

bool NeuroNet::NeuroNet::ResizeLayer(int pLayerIndex, int pLayerSize)
{
	if (pLayerIndex >= this->NeuroNetVector.size())
		return false;
	if (pLayerIndex > 0)
		this->NeuroNetVector[pLayerIndex].ResizeLayer(this->NeuroNetVector[pLayerIndex - 1].LayerSize(), pLayerSize);
	else
		this->NeuroNetVector[pLayerIndex].ResizeLayer(this->InputSize, pLayerSize);
	if (pLayerIndex + 1 < this->NeuroNetVector.size())
		this->NeuroNetVector[pLayerIndex + 1].ResizeLayer(pLayerSize, this->NeuroNetVector[pLayerIndex + 1].LayerSize());
	return true;
}

void NeuroNet::NeuroNet::SetInputSize(int pInputSize)
{
	this->InputSize = pInputSize;
	this->NeuroNetVector[0].ResizeLayer(pInputSize, this->NeuroNetVector[0].LayerSize());
}

void NeuroNet::NeuroNet::ResizeNeuroNet(int pLayerCount)
{
	this->NeuroNetVector.resize(pLayerCount);
}

bool NeuroNet::NeuroNet::SetInput(Matrix::Matrix<float> pInputMatrix)
{
	if (this->NeuroNetVector.size() > 0)
		return this->NeuroNetVector[0].SetInput(pInputMatrix);
	else
		return false;
}

Matrix::Matrix<float> NeuroNet::NeuroNet::GetOutput()
{
	this->NeuroNetVector[0].CalculateOutput();
	for (int i = 1; i < this->LayerCount; i++)
	{
		this->NeuroNetVector[i].SetInput(this->NeuroNetVector[i - 1].ReturnOutputMatrix());
		this->NeuroNetVector[i].CalculateOutput();
	}
	return this->NeuroNetVector[this->LayerCount - 1].ReturnOutputMatrix();
}
