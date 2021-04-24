/**
 * @file neuronet.h
 * @author Jacob Borden (amenra.beats@@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once
#ifdef NEURONET_DLL_EXPORTS
#define NEURONET_DLL_API __declspec(dllexport)
#else
#define NEURONET_DLL_API __declspec(dllimport)
#endif // NEURONET_DLL_EXPORTS

namespace neuronet
{
    /**
     * @brief NeuroNet is a library for creating neuro networks for artificial intelligence and machine learning.
     * 
     * 
     * 
     */
    extern "C++" NEURONET_DLL_API class NeuroNet
    {

    };
}