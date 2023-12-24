package utils

import (
	"math"

	"github.com/elvin-mark/goLLM/data"
)

func Sigmoid(x float32) (t float32) {
	return 1 / (1 + data.Exp(x))
}

func GeLU(x float32) (t float32) {
	return 0.5 * x * (1 + data.Tanh(data.Sqrt(2/math.Pi)*(x+0.044715*(x*x*x))))
}

func ReLU(x float32) (t float32) {
	return data.Max(x, 0)
}
