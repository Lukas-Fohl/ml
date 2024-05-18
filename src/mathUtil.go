package main

import "math"

func reLu(input float64) float64 {
	if input > 0.0 {
		return input
	}
	return 0.0
}

func sigmoid(input float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -input/5))
}
