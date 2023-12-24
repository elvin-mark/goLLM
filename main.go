package main

import (
	"fmt"
	"os"

	"github.com/elvin-mark/goLLM/models"
)

func gpt(prompt string) {
	fmt.Println("Loading tokenizer ...")
	encoder := models.NewEncoder(os.Getenv("GPT2_TOKENIZER_PATH"))
	fmt.Println("Encoding prompt ...")
	tokens := encoder.Encode(prompt)
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

func rwkv(prompt string) {
	fmt.Println("Loading tokenizer ...")
	encoder := models.NewEncoder(os.Getenv("RWKV_TOKENIZER_PATH"))
	fmt.Println("Encoding prompt ...")
	tokens := encoder.Encode(prompt)
	fmt.Println("prompt tokens: ", tokens)

	fmt.Println("Creating model ...")
	conf := models.NewRWKVConfig()
	rwkv := models.NewRWKV(conf)
	fmt.Println("Initializing model ...")
	rwkv.Init()
	fmt.Println("Loading weights")
	rwkv.LoadWeights(os.Getenv("RWKV_CONVERTED_MODEL_PATH"))
	fmt.Println("Generating tokens ...")
	seq := rwkv.Generate(tokens, 10)
	fmt.Println("generated tokens: ", seq)
	fmt.Println(encoder.Decode(seq))
}

func main() {
	// gpt(os.Args[1])
	rwkv(os.Args[1])
}
