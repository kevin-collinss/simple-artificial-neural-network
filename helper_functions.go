package main

import (
	"gonum.org/v1/gonum/mat"
)

func Dot(m, n mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	_, cols := n.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Product(m, n)
	return output
}

func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Apply(fn, m)
	return output
}

func Scale(scalar float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Scale(scalar, m)
	return output
}

func Multiply(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.MulElem(m, n)
	return output
}

func Add(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Add(m, n)
	return output
}

func Subtract(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Sub(m, n)
	return output
}

func AddScalar(i float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	linear := make([]float64, rows*cols)
	for x := 0; x < len(linear); x++ {
		linear[x] = i
	}

	n := mat.NewDense(rows, cols, linear)
	return Add(m, n)
}
