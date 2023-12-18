package utils

import "math"

func Sigmoid(x float64) (t float64) {
	return 1 / (1 + math.Exp(x))
}

func GeLU(x float64) (t float64) {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

func ReLU(x float64) (t float64) {
	return math.Max(x, 0)
}
