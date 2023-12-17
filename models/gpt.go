package models

import (
	"github.com/elvin-mark/goLLM/data"
	"github.com/elvin-mark/goLLM/utils"
)

type Param struct {
	w *data.Tensor
	b *data.Tensor
}

type MLP struct {
	fc   Param
	proj Param
}

type Attention struct {
	attn Param
	proj Param
}

type Block struct {
	ln1  Param
	attn Attention
	ln2  Param
	mlp  MLP
}

type GPT2Config struct {
	nHeads int
}

type GPT2 struct {
	conf   GPT2Config
	wpe    Param
	wte    Param
	blocks []Block
	lnF    Param
}

func attention(q *data.Tensor, k *data.Tensor, v *data.Tensor, mask *data.Tensor) (r data.Tensor) {

	return
}

func ffn(x *data.Tensor, mlp MLP) (r data.Tensor) {
	return
}

func mha(x *data.Tensor, attn Attention, conf GPT2Config) (r data.Tensor) {
	return
}

func transformerBlock(x *data.Tensor, block Block, conf GPT2Config) data.Tensor {
	o := utils.LayerNorm(x, block.ln1.w, block.ln1.b)
	o = mha(&o, block.attn, conf)
	o = o.Add(x)

	r := utils.LayerNorm(&o, block.ln2.w, block.ln2.b)
	r = ffn(&r, block.mlp)
	r = r.Add(&o)
	return r
}

func (g *GPT2) forward(tokens []int) data.Tensor {
	l := make([]int, len(tokens))
	for i, _ := range tokens {
		l[i] = i
	}
	te := g.wte.w.SubTensor(tokens)
	pe := g.wpe.w.SubTensor(l)
	x := te.Add(&pe)
	for _, block := range g.blocks {
		x = transformerBlock(&x, block, g.conf)
	}
	x = utils.LayerNorm(&x, g.lnF.w, g.lnF.b)
	return x
}

func NewGPT2Config() GPT2Config {
	return GPT2Config{
		nHeads: 12,
	}
}
func NewGPT2(conf GPT2Config) GPT2 {
	return GPT2{
		conf: conf,
	}
}
