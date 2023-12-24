package data

import (
	"math"
	"math/rand"
)

type Tensor struct {
	Data  [][]float32
	Shape []int
}

func NewTensor(data [][]float32, shape []int) (r Tensor) {
	if data == nil {
		data = make([][]float32, shape[0])
		for i := 0; i < shape[0]; i++ {
			data[i] = make([]float32, shape[1])
		}
	}

	r = Tensor{
		Data:  data,
		Shape: shape,
	}
	return
}

func (t Tensor) Random() {
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			t.Data[i][j] = rand.Float32()
		}
	}
}

func (t Tensor) Fill(val float32) {
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			t.Data[i][j] = val
		}
	}
}

func (t Tensor) SubTensor(pos []int) (r Tensor) {
	r = NewTensor(nil, []int{len(pos), t.Shape[1]})
	for i, val := range pos {
		r.Data[i] = t.Data[val]
	}
	return
}

func (t Tensor) Transpose() (r Tensor) {
	r = NewTensor(nil, []int{t.Shape[1], t.Shape[0]})
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			r.Data[j][i] = t.Data[i][j]
		}
	}
	return
}

func (t Tensor) UpTri() {
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			if j > i {
				t.Data[i][j] = 1.
			} else {
				t.Data[i][j] = 0.
			}
		}
	}
}

func (t Tensor) DownTri() {
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			if j > i {
				t.Data[i][j] = 0.
			} else {
				t.Data[i][j] = 1.
			}
		}
	}
}

func (t Tensor) Add(s Tensor) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	bx := 1
	if s.Shape[0] == 1 {
		bx = 0
	}
	by := 1
	if s.Shape[1] == 1 {
		by = 0
	}
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			r.Data[i][j] = t.Data[i][j] + s.Data[i*bx][j*by]
		}
	}
	return
}

func (t Tensor) Sub(s Tensor) (r Tensor) {
	bx := 1
	if s.Shape[0] == 1 {
		bx = 0
	}
	by := 1
	if s.Shape[1] == 1 {
		by = 0
	}
	r = NewTensor(nil, t.Shape)
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			r.Data[i][j] = t.Data[i][j] - s.Data[i*bx][j*by]
		}
	}
	return
}

func (t Tensor) Mul(s Tensor) (r Tensor) {
	bx := 1
	if s.Shape[0] == 1 {
		bx = 0
	}
	by := 1
	if s.Shape[1] == 1 {
		by = 0
	}
	r = NewTensor(nil, t.Shape)
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			r.Data[i][j] = t.Data[i][j] * s.Data[i*bx][j*by]
		}
	}
	return
}

func (t Tensor) Div(s Tensor) (r Tensor) {
	bx := 1
	if s.Shape[0] == 1 {
		bx = 0
	}
	by := 1
	if s.Shape[1] == 1 {
		by = 0
	}
	r = NewTensor(nil, t.Shape)
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			r.Data[i][j] = t.Data[i][j] / s.Data[i*bx][j*by]
		}
	}
	return
}

func (t Tensor) Sum(axis int) (r Tensor) {
	if axis == 0 {
		r = NewTensor(nil, []int{1, t.Shape[1]})
		for i := 0; i < t.Shape[1]; i++ {
			s := float32(0.0)
			for j := 0; j < t.Shape[0]; j++ {
				s += t.Data[j][i]
			}
			r.Data[0][i] = s
		}
	} else if axis == 1 {
		r = NewTensor(nil, []int{t.Shape[0], 1})
		for i := 0; i < t.Shape[0]; i++ {
			s := float32(0.0)
			for j := 0; j < t.Shape[1]; j++ {
				s += t.Data[i][j]
			}
			r.Data[i][0] = s
		}
	} else {
		r = NewTensor(nil, []int{1, 1})
		s := float32(0.0)
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				s += t.Data[i][j]
			}
		}
		r.Data[0][0] = s
	}
	return
}

func (t Tensor) Mean(axis int) (r Tensor) {
	if axis == 0 {
		r = NewTensor(nil, []int{1, t.Shape[1]})
		for i := 0; i < t.Shape[1]; i++ {
			s := float32(0.0)
			for j := 0; j < t.Shape[0]; j++ {
				s += t.Data[j][i]
			}
			r.Data[0][i] = s / float32(t.Shape[0])
		}
	} else if axis == 1 {
		r = NewTensor(nil, []int{t.Shape[0], 1})
		for i := 0; i < t.Shape[0]; i++ {
			s := float32(0.0)
			for j := 0; j < t.Shape[1]; j++ {
				s += t.Data[i][j]
			}
			r.Data[i][0] = s / float32(t.Shape[1])
		}
	} else {
		r = NewTensor(nil, []int{1, 1})
		s := float32(0.0)
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				s += t.Data[i][j]
			}
		}
		r.Data[0][0] = s / (float32(t.Shape[0] * t.Shape[1]))
	}
	return
}

func (t Tensor) Var(axis int, u Tensor) (r Tensor) {
	r = NewTensor(nil, u.Shape)
	if axis == 0 {
		for i := 0; i < t.Shape[1]; i++ {
			s := float32(0.0)
			for j := 0; j < t.Shape[0]; j++ {
				tmp := t.Data[j][i] - u.Data[0][i]
				s += tmp * tmp
			}
			r.Data[0][i] = s / float32(t.Shape[0])
		}
	} else if axis == 1 {
		for i := 0; i < t.Shape[0]; i++ {
			s := float32(0)
			for j := 0; j < t.Shape[1]; j++ {
				tmp := t.Data[i][j] - u.Data[i][0]
				s += tmp * tmp
			}
			r.Data[i][0] = s / float32(t.Shape[1])
		}
	} else {
		r = NewTensor(nil, []int{1, 1})
		s := float32(0)
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				tmp := t.Data[i][j] - u.Data[0][0]
				s += tmp * tmp
			}
		}
		r.Data[0][0] = s / (float32(t.Shape[0] * t.Shape[1]))
	}
	return
}

func (t Tensor) Std(axis int, u Tensor) (r Tensor) {
	v := t.Var(axis, u)
	r = v.Apply(Sqrt)
	return
}

func (t Tensor) Max(axis int) (r Tensor) {
	if axis == 0 {
		r = NewTensor(nil, []int{1, t.Shape[1]})
		for i := 0; i < t.Shape[1]; i++ {
			s := float32(math.Inf(-1))
			for j := 0; j < t.Shape[0]; j++ {
				s = Max(t.Data[j][i], s)
			}
			r.Data[0][i] = s
		}
	} else if axis == 1 {
		r = NewTensor(nil, []int{t.Shape[0], 1})
		for i := 0; i < t.Shape[0]; i++ {
			s := float32(math.Inf(-1))
			for j := 0; j < t.Shape[1]; j++ {
				s = Max(t.Data[i][j], s)
			}
			r.Data[i][0] = s
		}
	}
	return
}

func (t Tensor) Dot(s Tensor) (r Tensor) {
	r = NewTensor(nil, []int{t.Shape[0], s.Shape[1]})
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < s.Shape[1]; j++ {
			acc := float32(0)
			for k := 0; k < t.Shape[1]; k++ {
				acc += t.Data[i][k] * s.Data[k][j]
			}
			r.Data[i][j] = acc
		}
	}

	return
}

func (t Tensor) Apply(fn func(float32) float32) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			r.Data[i][j] = fn(t.Data[i][j])
		}
	}
	return
}
