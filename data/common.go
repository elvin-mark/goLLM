package data

import "math"

func Exp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func Max(x, y float32) float32 {
	if x > y {
		return x
	}
	return y
}
