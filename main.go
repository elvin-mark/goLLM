package main

import (
	"fmt"
	"os"

	"github.com/elvin-mark/goLLM/models"
)

func main() {
	fmt.Println("Loading tokenizer ...")
	encoder := models.NewEncoder(os.Getenv("GPT2_TOKENIZER_PATH"))
	fmt.Println("Encoding prompt ...")
	tokens := encoder.Encode(os.Args[1])
	fmt.Println("prompt tokens: ", tokens)

	fmt.Println("Creating model ...")
	conf := models.NewGPT2Config()
	gpt := models.NewGPT2(conf)
	fmt.Println("Initializing model ...")
	gpt.Init()
	fmt.Println("Loading weights")
	gpt.LoadWeights(os.Getenv("GPT2_CONVERTED_MODEL_PATH"))
	fmt.Println("Generating tokens ...")
	seq := gpt.Generate(tokens, 10)
	fmt.Println("generated tokens: ", seq)
	fmt.Println(encoder.Decode(seq))
}
