package nueralnetwork

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	return
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func (net Network) Predict(inputData []float64) mat.Matrix {

	//creates a matrix with nuerons for each input put into rows and a single column
	inputs := mat.NewDense(len(inputData), 1, inputData)

	//the hidden inputs are calculated by getting the dot product of the weights and the inputs
	hiddenInputs := dot(net.hiddenWeights, inputs)

	//the outputs are gotten by apply the sigmoid activation function to the inputs
	hiddenOutputs := apply(sigmoid, hiddenInputs)

	//the same dot product operation is done on the outputs
	finalInputs := dot(net.outputWeights, hiddenOutputs)

	//finally the sigmoid function is applied to the outputs
	finalOutputs := apply(sigmoid, finalInputs)
	return finalOutputs
}

func (net *Network) Train(inputData []float64, targetData []float64) {

	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	//get the errors
	targets := mat.NewDense(len(targetData), 1, targetData)

	//subtracts each target value from the final outputs
	outputErrors := subtract(targets, finalOutputs)

	//gets the errors of the weights by getting the dot product of the weights by the output error
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	//change the weights by scaling them at a rate following the formula:
	//	Δwjk = -l.(tk-ok)·ok(1-ok)·oj
	net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	//so the same for the hidden weights
	net.hiddenWeights = add(net.hiddenWeights,
		scale(net.learningRate,
			dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}
