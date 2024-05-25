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

	input     [4]float64
	expOutput [2]float64
}

func getContext() context {
	tempContext := context{}
	//values
	tempContext.playerPosX = 0.01 + rand.Float64()*(9.9-0.01)
	tempContext.playerPosY = 0.01 + rand.Float64()*(9.9-0.01)

	tempContext.pointPosX = 0.01 + rand.Float64()*(9.9-0.01)
	tempContext.pointPosY = 0.01 + rand.Float64()*(9.9-0.01)

	if tempContext.playerPosX > tempContext.pointPosX {
		tempContext.input[0] = 0.5
		tempContext.input[1] = 0.0
		tempContext.expOutput[0] = 0.6
	} else {
		tempContext.input[0] = 0.0
		tempContext.input[1] = 0.5
		tempContext.expOutput[0] = 0.4
	}

	if tempContext.playerPosY > tempContext.pointPosY {
		tempContext.input[2] = 0.5
		tempContext.input[3] = 0.0
		tempContext.expOutput[1] = 0.6
	} else {
		tempContext.input[2] = 0.0
		tempContext.input[3] = 0.5
		tempContext.expOutput[1] = 0.4
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
