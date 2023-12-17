package utils

import (
	"math"
	"strconv"

	"github.com/elvin-mark/goLLM/data"
)

func LayerNorm(x *data.Tensor, w *data.Tensor, b *data.Tensor) (t data.Tensor) {
	eps, _ := strconv.ParseFloat("1E-5", 64)
	fn := func(x float64) float64 {
		return math.Sqrt(x + eps)
	}
	u := x.Mean(-1)
	v := x.Var(-1, &u)
	s := v.Apply(fn)
	o := x.Sub(&u)
	o = o.Div(&s)
	o = o.Mul(w)
	t = o.Add(b)
	return
}

func Softmax(x *data.Tensor, axis int) (t data.Tensor) {
	m := x.Max(axis)
	o := x.Sub(&m)
	o = o.Apply(math.Exp)
	so := o.Sum(axis)
	t = o.Div(&so)
	return
}
