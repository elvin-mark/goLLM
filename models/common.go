package models

import "github.com/elvin-mark/goLLM/data"

type Param struct {
	w data.Tensor
	b data.Tensor
}

func SamplesProbs(probs []float64, temperature float64, topK int) int {
	return 0
}
