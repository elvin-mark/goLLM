package data

type Index struct {
	Curr      []int
	Shape     []int
	Stride    []int
	Offset    int
	Transpose []int
}

func NewIndex(shape []int) Index {
	acc := 1
	n := len(shape)
	stride := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		stride[i] = acc
		acc *= shape[i]
	}
	return Index{
		Curr:      make([]int, len(shape)),
		Shape:     shape,
		Stride:    stride,
		Offset:    0,
		Transpose: nil,
	}
}

func (idx *Index) Clone() (nIdx Index) {
	nIdx = NewIndex(idx.Shape)
	copy(nIdx.Curr, idx.Curr)
	copy(nIdx.Shape, idx.Shape)
	copy(nIdx.Stride, idx.Stride)
	nIdx.Offset = idx.Offset
	copy(nIdx.Transpose, idx.Transpose)
	return
}

func (idx *Index) Reset() {
	for i := 0; i < len(idx.Curr); i++ {
		idx.Curr[i] = 0
	}
}

func (idx *Index) Inc() bool {
	n := len(idx.Shape)
	carry := 1
	for i := n - 1; i >= 0; i-- {
		if carry != 0 {
			carry = (idx.Curr[i] + 1) / idx.Shape[i]
			idx.Curr[i] = (idx.Curr[i] + 1) % idx.Shape[i]
		}
	}
	return carry == 1
}

func (idx *Index) Broadcast(shape []int) {
	acc := 1
	n := len(shape)
	stride := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		if shape[i] != 1 {
			stride[i] = acc
		}
		acc *= shape[i]
	}
	idx.Stride = stride
}

func (idx *Index) GetPosition() int {
	p := 0
	for i, val := range idx.Stride {
		p += idx.Curr[i] * val
	}
	return p + idx.Offset
}

func (idx *Index) GetOffsetIndex(pos []int) (newIdx Index) {
	offset := 0
	shape := make([]int, 0)
	stride := make([]int, 0)
	for i, val := range pos {
		if val < 0 {
			shape = append(shape, idx.Shape[i])
			stride = append(stride, idx.Stride[i])
		} else {
			offset += idx.Stride[i] * val
		}
	}
	return Index{
		Curr:      make([]int, len(shape)),
		Shape:     shape,
		Stride:    stride,
		Offset:    offset,
		Transpose: nil,
	}
}
