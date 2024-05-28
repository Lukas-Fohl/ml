package main

import (
	"math"
	"math/rand"
)

/*
training:
	-simulate state
	-get expOutput --> run network
	-get cost
	-change some weight | fat bitch
how to:
	-go through all weights change bias --> change when wrong or right
*/

type context struct {
	playerPosX float64
	playerPosY float64

	pointPosX float64
	pointPosY float64

	input     []float64
	expOutput []float64
}

func getContext() context {
	tempContext := context{}
	//values
	tempContext.playerPosX = 0.01 + rand.Float64()*(9.9-0.01)
	tempContext.playerPosY = 0.01 + rand.Float64()*(9.9-0.01)

	tempContext.pointPosX = 0.01 + rand.Float64()*(9.9-0.01)
	tempContext.pointPosY = 0.01 + rand.Float64()*(9.9-0.01)

	rndBig := 0.6 + rand.Float64()*(0.9-0.6)
	rndSmall := 0.01 + rand.Float64()*(0.4-0.01)

	if tempContext.playerPosX > tempContext.pointPosX {
		tempContext.input = append(tempContext.input, rndBig)
		tempContext.input = append(tempContext.input, rndSmall)
		tempContext.expOutput = append(tempContext.expOutput, 0.6)
	} else {
		tempContext.input = append(tempContext.input, rndSmall)
		tempContext.input = append(tempContext.input, rndBig)
		tempContext.expOutput = append(tempContext.expOutput, 0.4)
	}

	if tempContext.playerPosY > tempContext.pointPosY {
		tempContext.input = append(tempContext.input, rndBig)
		tempContext.input = append(tempContext.input, rndSmall)
		tempContext.expOutput = append(tempContext.expOutput, 0.6)
	} else {
		tempContext.input = append(tempContext.input, rndSmall)
		tempContext.input = append(tempContext.input, rndBig)
		tempContext.expOutput = append(tempContext.expOutput, 0.4)
	}

	return tempContext
}

func cost(contextInput context, output []float64) float64 {
	//0 < cost < 1 --> bigger = worse
	tempCost := 0.0
	tempCost += math.Abs(contextInput.expOutput[0] - output[0])
	tempCost += math.Abs(contextInput.expOutput[1] - output[1])
	return tempCost
}

func training(networkIn network) network {
	stepSize := 0.01

	for layerIter := 0; layerIter < len(networkIn.config)-1; layerIter++ {
		for weightIter := 0; weightIter < len(networkIn.layers[layerIter].weights); weightIter++ {
			tempContext := getContext()
			//get original cost
			tempOutput, err := run(networkIn, tempContext.input)
			if err != nil {
				panic(err)
			}
			ogCost := cost(tempContext, tempOutput)

			//+stepsize --> cost
			networkIn.layers[layerIter].weights[weightIter].value += stepSize
			tempOutput, err = run(networkIn, tempContext.input)
			if err != nil {
				panic(err)
			}
			costP := cost(tempContext, tempOutput)

			//-stepsize --> cost
			networkIn.layers[layerIter].weights[weightIter].value -= 2 * stepSize
			tempOutput, err = run(networkIn, tempContext.input)
			if err != nil {
				panic(err)
			}
			costM := cost(tempContext, tempOutput)

			//set to last value
			networkIn.layers[layerIter].weights[weightIter].value += stepSize

			if costM < ogCost || costP < ogCost {
				if costM < costP {
					networkIn.layers[layerIter].weights[weightIter].value -= stepSize
					continue
				} else {
					networkIn.layers[layerIter].weights[weightIter].value += stepSize
					continue
				}
			}
		}

		for biasIter := 0; biasIter < len(networkIn.layers[layerIter].biases); biasIter++ {
			tempContext := getContext()
			//get original cost
			tempOutput, err := run(networkIn, tempContext.input)
			if err != nil {
				panic(err)
			}
			ogCost := cost(tempContext, tempOutput)

			//+stepsize --> cost
			networkIn.layers[layerIter].biases[biasIter].value += stepSize
			tempOutput, err = run(networkIn, tempContext.input)
			if err != nil {
				panic(err)
			}
			costP := cost(tempContext, tempOutput)

			//-stepsize --> cost
			networkIn.layers[layerIter].biases[biasIter].value -= 2 * stepSize
			tempOutput, err = run(networkIn, tempContext.input)
			if err != nil {
				panic(err)
			}
			costM := cost(tempContext, tempOutput)

			//set to last value
			networkIn.layers[layerIter].biases[biasIter].value += stepSize

			if costM < ogCost || costP < ogCost {
				if costM < costP {
					networkIn.layers[layerIter].biases[biasIter].value -= stepSize
					continue
				} else {
					networkIn.layers[layerIter].biases[biasIter].value += stepSize
					continue
				}
			}
		}
	}
	return networkIn
}
