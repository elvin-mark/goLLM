package data

import (
	"math"
	"math/rand"
)

type Tensor struct {
	Data  []float64
	Shape []int
	Idx   Index
}

func NewTensor(data []float64, shape []int) (r Tensor) {
	size := 1
	for _, i := range shape {
		size *= i
	}
	if data == nil {
		data = make([]float64, size)
	}

	r = Tensor{
		Data:  data,
		Shape: shape,
		Idx:   NewIndex(shape),
	}
	return
}

func (t *Tensor) At(i Index) float64 {
	p := i.GetPosition()
	return t.Data[p]
}

func (t *Tensor) Get(pos []int) (r Tensor) {
	newIdx := t.Idx.GetOffsetIndex(pos)
	r = NewTensor(t.Data, newIdx.Shape)
	r.Idx = newIdx
	return
}

func (t *Tensor) SubTensor(idxs []int) (r Tensor) {
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	shape[0] = len(idxs)
	r = NewTensor(nil, shape)
	idx1 := t.GetIndex()
	idx2 := r.GetIndex()
	idx1.Curr = idx2.Curr
	for {
		tmp := idx2.Curr[0]
		idx2.Curr[0] = idxs[idx2.Curr[0]]
		val := t.At(idx1)
		idx2.Curr[0] = tmp
		r.Data[idx2.GetPosition()] = val
		if idx2.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Random() {
	for i := 0; i < len(t.Data); i++ {
		t.Data[i] = rand.Float64()
	}
}

func (t *Tensor) Fill(val float64) {
	for i := 0; i < len(t.Data); i++ {
		t.Data[i] = val
	}
}

func (t *Tensor) GetIndex() Index {
	return t.Idx.Clone()
}

func (t *Tensor) Add(s *Tensor) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	idx1 := t.GetIndex()
	idx2 := t.GetIndex()
	idx2.Broadcast(s.Shape)
	for {
		r.Data[idx1.GetPosition()] = t.At(idx1) + s.At(idx2)
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Sub(s *Tensor) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	idx1 := t.GetIndex()
	idx2 := t.GetIndex()
	idx2.Broadcast(s.Shape)
	for {
		r.Data[idx1.GetPosition()] = t.At(idx1) - s.At(idx2)
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Mul(s *Tensor) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	idx1 := t.GetIndex()
	idx2 := t.GetIndex()
	idx2.Broadcast(s.Shape)
	for {
		r.Data[idx1.GetPosition()] = t.At(idx1) * s.At(idx2)
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Div(s *Tensor) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	idx1 := t.GetIndex()
	idx2 := t.GetIndex()
	idx2.Broadcast(s.Shape)
	for {
		r.Data[idx1.GetPosition()] = t.At(idx1) / s.At(idx2)
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Sum(axis []int) (r Tensor) {
	idx1 := t.GetIndex()
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	for _, i := range axis {
		shape[i] = 1
	}
	idx2 := t.GetIndex()
	idx2.Broadcast(shape)
	r = NewTensor(nil, shape)
	for {
		r.Data[idx2.GetPosition()] += t.At(idx1)
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Mean(axis []int) (r Tensor) {
	idx1 := t.GetIndex()
	shape := make([]int, len(t.Shape))
	acc := 1
	copy(shape, t.Shape)
	for _, i := range axis {
		acc *= shape[i]
		shape[i] = 1
	}
	M := float64(acc)
	idx2 := t.GetIndex()
	idx2.Broadcast(shape)
	r = NewTensor(nil, shape)
	for {
		r.Data[idx2.GetPosition()] += t.At(idx1) / M
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Var(axis []int, u *Tensor) (r Tensor) {
	if u == nil {
		tmp := t.Mean(axis)
		u = &tmp
	}
	acc := 1
	for _, i := range axis {
		acc *= t.Shape[i]
	}
	M := float64(acc)
	idx1 := t.GetIndex()
	idx2 := t.GetIndex()
	idx2.Broadcast(u.Shape)
	r = NewTensor(nil, u.Shape)
	for {
		r.Data[idx2.GetPosition()] += math.Pow(t.At(idx1)-u.At(idx2), 2) / M
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Std(axis []int, u *Tensor) (r Tensor) {
	v := t.Var(axis, u)
	r = v.Apply(math.Sqrt)
	return
}

func (t *Tensor) Max(axis []int) (r Tensor) {
	idx1 := t.GetIndex()
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	for _, i := range axis {
		shape[i] = 1
	}
	idx2 := t.GetIndex()
	idx2.Broadcast(shape)
	r = NewTensor(nil, shape)
	r.Fill(math.Inf(-1))
	for {
		r.Data[idx2.GetPosition()] = math.Max(r.Data[idx2.GetPosition()], t.At(idx1))
		idx2.Inc()
		if idx1.Inc() {
			break
		}
	}
	return
}

func (t *Tensor) Dot(s *Tensor) (r Tensor) {
	// idx1 := t.GetIndex()
	// idx2 := t.GetIndex()

	return
}

func (t *Tensor) Apply(fn func(float64) float64) (r Tensor) {
	r = NewTensor(nil, t.Shape)
	for i, val := range t.Data {
		r.Data[i] = fn(val)
	}
	return
}
