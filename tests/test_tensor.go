package tests

import (
	"fmt"

	"github.com/elvin-mark/goLLM/data"
)

func TestTensor() {
	sData := []float64{0.48306873, 0.38863715, 0.85980998, 0.3743831, 0.23139692, 0.45396354}
	tData := []float64{0.94483818, 0.37985632, 0.32079834, 0.51040672, 0.02194908, 0.09708848, 0.48933504, 0.34737211, 0.96558146, 0.61977999, 0.36555447, 0.90812078, 0.28409845, 0.18949074, 0.97783185, 0.261856, 0.25598924, 0.95570657, 0.80096669, 0.80904141, 0.00164683, 0.4092291, 0.38559746, 0.28392359}

	t := data.NewTensor(tData, []int{3, 4, 2})
	u := t.Mean([]int{1})
	fmt.Println("\nmean of t: ", u)
	fmt.Println("\nvar of t: ", t.Var([]int{1}, &u))
	fmt.Println("\nstd of t: ", t.Std([]int{1}, &u))
	s := data.NewTensor(sData, []int{3, 1, 2})

	a := t.Add(&s)
	fmt.Println("\nt + s: ", a)

	a = t.Sub(&s)
	fmt.Println("\nt - s: ", a)

	a = t.Mul(&s)
	fmt.Println("\nt * s: ", a)

	a = t.Div(&s)
	fmt.Println("\nt / s: ", a)

	fmt.Println("\nmax of t: ", t.Max([]int{1}))

	fmt.Println("\nsum of t: ", t.Sum([]int{1}))

	tmp := t.Get([]int{-1, 2, -1})
	idx := tmp.GetIndex()
	for {
		fmt.Println(tmp.At(idx))
		if idx.Inc() {
			break
		}
	}

	x := data.NewTensor([]float64{1., 2., 3., 4., 5., 6., 7., 8., 9., 10}, []int{5, 2})
	fmt.Println(x.SubTensor([]int{1, 3}))
}
