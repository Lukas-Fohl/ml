package main

import "fmt"

func main() {
	config := []int{4, 8, 2}

	var n network

	n, err := networkBuild(config)
	if err != nil {
		panic(err)
	}

	inputValues := []float64{0.5, 0.5, 0.5, 0.5}

	output, err := run(n, inputValues)
	if err != nil {
		panic(err)
	}

	fmt.Println(output)

	tn := n
	for i := 0; i < 100000; i++ {
		tn = training(tn)
	}

	outputTr, err := run(tn, inputValues)
	if err != nil {
		panic(err)
	}

	fmt.Println(outputTr)

	return
}
