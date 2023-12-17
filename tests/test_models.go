package tests

import (
	"fmt"

	"github.com/elvin-mark/goLLM/models"
)

func TestModels() {
	conf := models.NewGPT2Config()
	gpt := models.NewGPT2(conf)
	fmt.Println(gpt)
}
