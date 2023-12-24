package utils

import (
	"strconv"

	"github.com/elvin-mark/goLLM/data"
)

func LayerNorm(x data.Tensor, w data.Tensor, b data.Tensor) (t data.Tensor) {
	eps64, _ := strconv.ParseFloat("1E-5", 32)
	eps := float32(eps64)
	fn := func(x float32) float32 {
		return data.Sqrt(x + eps)
	}
	u := x.Mean(1)
	v := x.Var(1, u)
	s := v.Apply(fn)
	return x.Sub(u).Div(s).Mul(w).Add(b)
}

func Softmax(x data.Tensor, axis int) (t data.Tensor) {
	o := x.Sub(x.Max(axis)).Apply(data.Exp)
	so := o.Sum(axis)
	return o.Div(so)
}

func Linear(x data.Tensor, w data.Tensor, b data.Tensor) (r data.Tensor) {
	return x.Dot(w).Add(b)
}
