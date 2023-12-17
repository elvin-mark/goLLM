package utils

import (
	"math"
	"strconv"

	"github.com/elvin-mark/goLLM/data"
)

func LayerNorm(x *data.Tensor, w *data.Tensor, b *data.Tensor) (t data.Tensor) {
	eps, _ := strconv.ParseFloat("1E-5", 64)
	axis := make([]int, len(x.Shape))
	for i := 0; i < len(x.Shape); i++ {
		axis[i] = i
	}
	fn := func(x float64) float64 {
		return math.Sqrt(x + eps)
	}
	u := x.Mean(axis)
	v := x.Var(axis, &u)
	s := v.Apply(fn)
	o := x.Sub(&u)
	o = o.Div(&s)
	o = o.Mul(w)
	t = o.Add(b)
	return
}

func Softmax(x *data.Tensor, axis []int) (t data.Tensor) {
	m := x.Max(axis)
	o := x.Sub(&m)
	o = o.Apply(math.Exp)
	so := o.Sum(axis)
	t = o.Div(&so)
	return
}
