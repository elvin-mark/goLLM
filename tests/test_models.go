package tests

import (
	"fmt"
	"os"

	"github.com/elvin-mark/goLLM/models"
)

func TestGPT2Model() {
	encoder := models.NewEncoder(os.Getenv("GPT2_TOKENIZER_PATH"))
	tokens := encoder.Encode("In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings.")
	fmt.Println("tokens: ", tokens)
	fmt.Println(encoder.Decode([]int{818, 11887, 11, 4731, 4583, 318, 257, 16200, 9355, 287, 543, 262, 966, 12, 2339, 13166, 286, 18758, 11887, 389, 6928, 416, 530, 12, 19577, 5563, 1444, 13042, 13}))
	conf := models.NewGPT2Config()
	gpt := models.NewGPT2(conf)
	gpt.Init()
	gpt.LoadWeights(os.Getenv("GPT2_CONVERTED_MODEL_PATH"))
	seq := gpt.Generate(tokens, 10)
	fmt.Println(seq)
	fmt.Println(encoder.Decode(seq))
}

func TestRWKVModel() {
	encoder := models.NewEncoder(os.Getenv("RWKV_TOKENIZER_PATH"))
	tokens := encoder.Encode("\nIn physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings.")
	fmt.Println("tokens: ", tokens)
	fmt.Println(encoder.Decode([]int{187, 688, 12057, 13, 2876, 3762, 310, 247, 10527, 7792, 275, 534, 253, 1127, 14, 3022, 6353, 273, 8091, 12057, 403, 7932, 407, 581, 14, 6967, 5113, 1925, 11559, 15}))
	conf := models.NewRWKVConfig()
	rwkv := models.NewRWKV(conf)
	rwkv.Init()
	rwkv.LoadWeights(os.Getenv("RWKV_CONVERTED_MODEL_PATH"))
	seq := rwkv.Generate(tokens, 20)
	fmt.Println(seq)
	fmt.Println(encoder.Decode(seq))
}

func TestModels() {
	// TestGPT2Model()
	TestRWKVModel()
}
