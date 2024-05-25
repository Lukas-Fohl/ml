package main

import (
	"fmt"
	"math/rand"
	"time"
)

type layerTypes int

const (
	hiddenType layerTypes = iota
	inputType  layerTypes = iota
	outputType layerTypes = iota
)

type weight struct {
	value float64
}

type bias struct {
	tempValue float64
	value     float64
}

type layer struct {
	layerType layerTypes
	biases    []bias
	weights   []weight
}

type network struct {
	config []int //0:input, 0..n-1:hidden, n:out
	layers []layer
}

func setupWeight(input float64) weight {
	return weight{value: input}
}

func setupBias(input float64) bias {
	return bias{value: input, tempValue: 0.0}
}

func networkBuild(input []int) (network, error) {
	if len(input) < 3 {
		return network{}, fmt.Errorf("[ERROR]network build error: wrong input size")
	}

	var n network

	n.config = input
	n.layers = []layer{}

	rand.Seed(time.Now().UnixNano())
	defaultBias := bias{value: 0.001 + rand.Float64()*(0.9-0.001), tempValue: 0.0}
	defaultWeight := weight{value: 0.001 + rand.Float64()*(0.9-0.001)}
	inputLayerSize := input[0]
	outputLayerSize := input[len(input)-1]
	hiddenLayerAm := len(input) - 2

	//input weights:
	inputBiases := []bias{}
	for i := 0; i < inputLayerSize; i++ {
		inputBiases = append(inputBiases, defaultBias)
	}
	inputWeightAm := input[1] * input[0]
	inputWeights := []weight{}
	for i := 0; i < inputWeightAm; i++ {
		inputWeights = append(inputWeights, defaultWeight)
	}
	inputLayer := layer{layerType: inputType, biases: inputBiases, weights: inputWeights}
	n.layers = append(n.layers, inputLayer)

	//hidden:
	for hiddenIdx := 0; hiddenIdx < hiddenLayerAm; hiddenIdx++ {
		hiddenBiases := []bias{}
		for i := 0; i < input[hiddenIdx+1]; i++ {
			hiddenBiases = append(hiddenBiases, bias{value: 0.01 + rand.Float64()*(0.50-0.01), tempValue: 0.0})
		}
		hiddenWeightAm := input[hiddenIdx+1] * input[hiddenIdx+2]
		hiddenWeights := []weight{}
		for i := 0; i < hiddenWeightAm; i++ {
			hiddenWeights = append(hiddenWeights, weight{value: 0.001 + rand.Float64()*(0.9-0.001)})
		}
		hiddenLayer := layer{layerType: hiddenType, biases: hiddenBiases, weights: hiddenWeights}
		n.layers = append(n.layers, hiddenLayer)
	}

	//output:
	outputBiases := []bias{}
	for i := 0; i < outputLayerSize; i++ {
		outputBiases = append(outputBiases, defaultBias)
	}
	outputLayer := layer{layerType: outputType, biases: outputBiases, weights: nil}
	n.layers = append(n.layers, outputLayer)

	return n, nil
}

func run(n network, input []float64) ([]float64, error) {
	if len(input) != len(n.layers[0].biases) {
		return input, fmt.Errorf("[ERROR]network run error: wrong input size")
	}
	//fill input:
	for i := 0; i < len(input); i++ {
		n.layers[0].biases[i].tempValue = input[i]
	}

	layerRunAm := len(n.config) - 2
	layerIdx := 1
	for p := 0; p < layerRunAm+1; p++ {
		for i := 0; i < len(n.layers[layerIdx].biases); i++ {
			sum := 0.0
			for j := 0; j < len(n.layers[layerIdx-1].biases); j++ {
				thisBiasAm := len(n.layers[layerIdx].biases)
				weightIdx := thisBiasAm*j + i
				sum += (n.layers[layerIdx-1].biases[j].tempValue * n.layers[layerIdx-1].weights[weightIdx].value) - n.layers[layerIdx-1].biases[j].value
			}
			n.layers[layerIdx].biases[i].tempValue = sigmoid(float64(sum))
		}
		layerIdx++
	}

	output := []float64{}
	for i := 0; i < len(n.layers[len(n.config)-1].biases); i++ {
		output = append(output, n.layers[len(n.config)-1].biases[i].tempValue)
	}

	return output, nil
}
