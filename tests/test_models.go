package tests

import (
	"fmt"

	"github.com/elvin-mark/goLLM/models"
)

func TestModels() {
	conf := models.NewGPT2Config()
	gpt := models.NewGPT2(conf)
	gpt.Init()
	o := gpt.Forward([]int{3, 4, 5})
	fmt.Println(o)
}
