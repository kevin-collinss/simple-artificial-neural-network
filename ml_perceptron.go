package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"

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

	//the hidden inputs are calculated by getting the Dot product of the weights and the inputs
	hiddenInputs := Dot(net.hiddenWeights, inputs)

	//the outputs are gotten by Apply the sigmoid activation function to the inputs
	hiddenOutputs := Apply(sigmoid, hiddenInputs)

	//the same Dot product operation is done on the outputs
	finalInputs := Dot(net.outputWeights, hiddenOutputs)

	//finally the sigmoid function is applied to the outputs
	finalOutputs := Apply(sigmoid, finalInputs)
	return finalOutputs
}

func (net *Network) Train(inputData []float64, targetData []float64) {

	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := Dot(net.hiddenWeights, inputs)
	hiddenOutputs := Apply(sigmoid, hiddenInputs)
	finalInputs := Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := Apply(sigmoid, finalInputs)

	//get the errors
	targets := mat.NewDense(len(targetData), 1, targetData)

	//Subtracts each target value from the final outputs
	outputErrors := Subtract(targets, finalOutputs)

	//gets the errors of the weights by getting the Dot product of the weights by the output error
	hiddenErrors := Dot(net.outputWeights.T(), outputErrors)

	//change the weights by scaling them at a rate following the formula:
	//	Δwjk = -l.(tk-ok)·ok(1-ok)·oj
	net.outputWeights = Add(net.outputWeights,
		Scale(net.learningRate,
			Dot(Multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	//so the same for the hidden weights
	net.hiddenWeights = Add(net.hiddenWeights,
		Scale(net.learningRate,
			Dot(Multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return Multiply(m, Subtract(ones, m)) // m * (1 - m)
}

func save(net Network) {
	h, err := os.Create("data/outputweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/outputweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

func load(net *Network) {
	h, err := os.Open("data/hiddenweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/outputweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}

func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// predict a number from an image
// image should be 28 x 28 PNG file
func predictFromImage(net Network, path string) int {
	input := dataFromImage(path)
	output := net.Predict(input)
	matrixPrint(output)
	best := 0
	highest := 0.0
	for i := 0; i < net.outputs; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}

func dataFromImage(filePath string) (pixels []float64) {
	// read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	// create a grayscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	// make a pixel array
	pixels = make([]float64, len(gray.Pix))
	// populate the pixel array subtract Pix from 255 because
	// that's how the MNIST database was trained (in reverse)
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.99) + 0.01
	}
	return
}
