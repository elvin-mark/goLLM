package tests

import (
	"fmt"

	"github.com/elvin-mark/goLLM/data"
	"github.com/elvin-mark/goLLM/utils"
)

func TestLayers() {
	tData := [][]float32{{0.94483818, 0.37985632, 0.32079834, 0.51040672, 0.02194908, 0.09708848}, {0.48933504, 0.34737211, 0.96558146, 0.61977999, 0.36555447, 0.90812078}, {0.28409845, 0.18949074, 0.97783185, 0.261856, 0.25598924, 0.95570657}, {0.80096669, 0.80904141, 0.00164683, 0.4092291, 0.38559746, 0.28392359}}

	wData := [][]float32{{0.26710264, 0.00996463, 0.53860701, 0.92964336, 0.06609951,
		0.53344476}, {0.67126192, 0.53982978, 0.44000614, 0.77342033,
		0.99382666, 0.49356887}, {0.02403988, 0.49437984, 0.54475234,
		0.18748107, 0.49779492, 0.29986873}, {0.81406694, 0.25246795,
		0.53746101, 0.97142713, 0.15901197, 0.75983879}}
	bData := [][]float32{{0.09420536, 0.15030868, 0.43614762, 0.40891283, 0.43668941,
		0.80184444}}

	t := data.NewTensor(tData, []int{4, 6})
	w := data.NewTensor(wData, []int{4, 6})
	b := data.NewTensor(bData, []int{1, 6})

	fmt.Println("layer norm: ", utils.LayerNorm(t, w, b))

	fmt.Println("softmax of t: ", utils.Softmax(t, 1))
}
