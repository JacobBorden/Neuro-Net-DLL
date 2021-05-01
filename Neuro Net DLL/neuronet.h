/**
 * @file neuronet.h
 * @author Jacob Borden (amenra.beats@@gmail.com)
 * @brief  Header file for neuronet.dll
 * @version 0.1
 * @date 2021-04-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */

//Header declearations
#pragma once
#ifdef NEURONET_DLL_EXPORTS
#define NEURONET_DLL_API __declspec(dllexport)
#else
#define NEURONET_DLL_API __declspec(dllimport)
#endif // NEURONET_DLL_EXPORTS

#include "../includes/matrix.h"
#include <vector>

namespace NeuroNet
{
	extern "C++" struct NEURONET_DLL_API LayerWeights
	{
		int WeightCount = 0;
		std::vector<BYTE> WeightsVector;
	};

	extern "C++" struct NEURONET_DLL_API LayerBiases
	{
		int BiasCount = 0;
		std::vector<BYTE> BiasVector;
	};

	/**
 * @brief The NeuroNetLayer class is an object that represents each individual layer of the neuro network
 * 
 * 
 */
	extern "C++" class NEURONET_DLL_API NeuroNetLayer
	{
	public:
		NeuroNetLayer();
		~NeuroNetLayer();
		void ResizeLayer(int pInputSize, int pLayerSize);
		Matrix::Matrix<float> CalculateOutput();
		bool SetInput(Matrix::Matrix<float> pInputMatrix);
		int WeightCount();
		int BiasCount();
		bool SetWeights(LayerWeights pWeights);
		bool SetBiases(LayerBiases pBiases);

	private:
		int LayerSize = 0;
		int InputSize = 0;
		Matrix::Matrix<float> InputMatrix;
		Matrix::Matrix<float> WeightMatrix;
		Matrix::Matrix<float> BiasMatrix;
		Matrix::Matrix<float> OutputMatrix;
		LayerWeights Weights;
		LayerBiases Biases;
	};

	

	/**
	 * @brief NeuroNet is a library for creating neuro networks for artificial intelligence and machine learning.
	 *
	 *
	 *
	 */
	extern "C++" class NEURONET_DLL_API NeuroNet
	{
	public:
		NeuroNet();
		NeuroNet(int pLayerCount);
		~NeuroNet();
		bool ResizeLayer(int pLayerIndex, int pLayerSize);
		void SetInputSize(int pInputSize);
		void ResizeNeuroNet(int pLayerCount);
		bool SetInput(int pInputSize);
		Matrix::Matrix<float> GetOutput();

	private:
		int InputSize = 0;
		int LayerCount = 0;
		std::vector<NeuroNetLayer> NeuroNetVector;
	};
}