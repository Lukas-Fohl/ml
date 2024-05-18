package main

import "fmt"

func main() {
	config := []int{4, 16, 8, 4}

	var n network

	n, err := networkBuild(config)
	if err != nil {
		panic(err)
	}

	inputValues := []float64{0.1, 0.5, 0.6, 0.1}

	output, err := run(n, inputValues)
	if err != nil {
		panic(err)
	}

	fmt.Println(output)
	return
}
