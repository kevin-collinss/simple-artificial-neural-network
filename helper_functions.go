package nueralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

func dot(m, n mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	_, cols := n.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Product(m, n)
	return output
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Apply(fn, m)
	return output
}

func scale(scalar float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Scale(scalar, m)
	return output
}

func multiply(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.MulElem(m, n)
	return output
}

func add(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Add(m, n)
	return output
}

func subtract(m, n mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	output := mat.NewDense(rows, cols, nil)
	output.Sub(m, n)
	return output
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	linear := make([]float64, rows*cols)
	for x := 0; x < len(linear); x++ {
		linear[x] = i
	}

	n := mat.NewDense(rows, cols, linear)
	return add(m, n)
}
