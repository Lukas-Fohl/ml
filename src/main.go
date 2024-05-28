package main

import "fmt"

func main() {
	config := []int{4, 8, 2}

	var n network

	n, err := networkBuild(config)
	if err != nil {
		panic(err)
	}

	tempContext := getContext()
	inputValues := tempContext.input

	output, err := run(n, inputValues)
	if err != nil {
		panic(err)
	}

	fmt.Print("input:\t\t\t")
	fmt.Println(inputValues)
	fmt.Println()

	fmt.Print("untrained output:\t")
	fmt.Println(output)
	fmt.Print("cost:\t\t\t")
	fmt.Println(fmt.Sprintln(cost(tempContext, output)))
	fmt.Println()

	tn := n
	for i := 0; i < 100000; i++ { //deep learning
		tn = training(tn)
	}

	output, err = run(tn, inputValues)
	if err != nil {
		panic(err)
	}
	fmt.Print("trained output:\t\t")
	fmt.Println(output)
	fmt.Print("cost:\t\t\t")
	fmt.Println(fmt.Sprintln(cost(tempContext, output)))

	return
}
