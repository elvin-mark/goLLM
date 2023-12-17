package tests

import (
	"fmt"

	"github.com/elvin-mark/goLLM/data"
)

func TestIndex() {
	idx := data.NewIndex([]int{3, 4})
	for {
		fmt.Println(idx, idx.GetPosition())
		if idx.Inc() {
			break
		}
	}
}
