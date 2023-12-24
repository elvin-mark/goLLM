package models

import (
	"bytes"
	"os"

	"github.com/elvin-mark/goLLM/data"
	"github.com/elvin-mark/goLLM/utils"
)

type RWKVAttention struct {
	timeDecay  Param
	timeFirst  Param
	timeMixK   Param
	timeMixV   Param
	timeMixR   Param
	key        Param
	value      Param
	receptance Param
	output     Param
}

type RWKVFFN struct {
	timeMixK   Param
	timeMixR   Param
	key        Param
	value      Param
	receptance Param
}

type RWKVBlock struct {
	ln1  Param
	attn RWKVAttention
	ln2  Param
	fnn  RWKVFFN
}

type RWKVConfig struct {
	nLayers   int
	embDim    int
	ctxLen    int
	vocabSize int
}

type RWKV struct {
	conf   RWKVConfig
	emb    Param
	ln0    Param
	blocks []RWKVBlock
	lnF    Param
	head   Param
}

func helperFn1(inp float32) float32 {
	return 1 - inp
}

func helperFn2(inp float32) float32 {
	return -data.Exp(inp)
}

func square(inp float32) float32 {
	return inp * inp
}

func timeMixing(x, lastX, lastNum, lastDen data.Tensor, attn RWKVAttention) (o, oldX, newNum, newDen data.Tensor) {
	k := attn.key.w.Dot(x.Mul(attn.timeMixK.w).Add(lastX.Mul(attn.timeMixK.w.Apply(helperFn1))))
	v := attn.value.w.Dot(x.Mul(attn.timeMixV.w).Add(lastX.Mul(attn.timeMixV.w.Apply(helperFn1))))
	r := attn.receptance.w.Dot(x.Mul(attn.timeMixR.w).Add(lastX.Mul(attn.timeMixR.w.Apply(helperFn1))))

	num := lastNum.Add(attn.timeFirst.w.Add(k).Apply(data.Exp).Mul(v))
	den := lastDen.Add(attn.timeFirst.w.Add(k).Apply(data.Exp))
	wkv := num.Div(den)

	rwkv := r.Apply(utils.Sigmoid).Mul(wkv)
	num = attn.timeDecay.w.Apply(helperFn2).Apply(data.Exp).Mul(lastNum).Add(k.Apply(data.Exp).Mul(v))
	den = attn.timeDecay.w.Apply(helperFn2).Apply(data.Exp).Mul(lastDen).Add(k.Apply(data.Exp))
	return attn.output.w.Dot(rwkv), x, num, den
}

func channelMixing(x, lastX data.Tensor, ffn RWKVFFN) (o, oldX data.Tensor) {
	k := ffn.key.w.Dot(x.Mul(ffn.timeMixK.w).Add(lastX.Mul(ffn.timeMixK.w.Apply(helperFn1))))
	r := ffn.receptance.w.Dot(x.Mul(ffn.timeMixR.w).Add(lastX.Mul(ffn.timeMixR.w.Apply(helperFn1))))

	vk := ffn.value.w.Dot(k.Apply(utils.ReLU).Apply(square))
	return r.Apply(utils.Sigmoid).Mul(vk), x
}

func NewRWKVConfig() RWKVConfig {
	return RWKVConfig{
		nLayers:   12,
		embDim:    768,
		vocabSize: 50277,
		ctxLen:    1024,
	}
}

func NewRWKV(conf RWKVConfig) RWKV {
	return RWKV{
		conf: conf,
	}
}

func (r *RWKV) Init() {
	{
		tmpW := data.NewTensor(nil, []int{r.conf.vocabSize, r.conf.embDim})
		tmpW.Random()
		r.emb = Param{
			w: tmpW,
			b: data.Tensor{},
		}
	}
	r.blocks = make([]RWKVBlock, r.conf.nLayers)
	for i := 0; i < r.conf.nLayers; i++ {
		tmpW := data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		tmpb := data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpb.Random()
		r.blocks[i].ln1 = Param{
			w: tmpW,
			b: tmpb,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].attn.timeDecay = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].attn.timeFirst = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].attn.timeMixK = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].attn.timeMixV = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].attn.timeMixR = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, r.conf.embDim})
		tmpW.Random()
		r.blocks[i].attn.key = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, r.conf.embDim})
		tmpW.Random()
		r.blocks[i].attn.value = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, r.conf.embDim})
		tmpW.Random()
		r.blocks[i].attn.receptance = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, r.conf.embDim})
		tmpW.Random()
		r.blocks[i].attn.output = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		tmpb = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpb.Random()

		r.blocks[i].ln2 = Param{
			w: tmpW,
			b: tmpb,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].fnn.timeMixK = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		r.blocks[i].fnn.timeMixR = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{3 * r.conf.ctxLen, r.conf.embDim})
		tmpW.Random()
		r.blocks[i].fnn.key = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, r.conf.embDim})
		tmpW.Random()
		r.blocks[i].fnn.receptance = Param{
			w: tmpW,
		}

		tmpW = data.NewTensor(nil, []int{r.conf.embDim, 3 * r.conf.ctxLen})
		tmpW.Random()
		r.blocks[i].fnn.value = Param{
			w: tmpW,
		}
	}
	{
		tmpW := data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		tmpb := data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpb.Random()
		r.ln0 = Param{
			w: tmpW,
			b: tmpb,
		}
	}

	{
		tmpW := data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpW.Random()
		tmpb := data.NewTensor(nil, []int{r.conf.embDim, 1})
		tmpb.Random()
		r.lnF = Param{
			w: tmpW,
			b: tmpb,
		}
	}

	{
		tmpW := data.NewTensor(nil, []int{r.conf.vocabSize, r.conf.embDim})
		tmpW.Random()
		r.head = Param{
			w: tmpW,
			b: data.Tensor{},
		}
	}
}

func (r RWKV) LoadWeights(modelPath string) {
	bs, err := os.ReadFile(modelPath)
	if err != nil {
		return
	}
	buf := bytes.NewReader(bs)

	LoadMatrix(buf, r.emb.w.Data, r.conf.vocabSize)
	LoadMatrix(buf, r.ln0.w.Data, r.conf.embDim)
	LoadMatrix(buf, r.ln0.b.Data, r.conf.embDim)
	for i := 0; i < r.conf.nLayers; i++ {
		LoadMatrix(buf, r.blocks[i].ln1.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].ln1.b.Data, r.conf.embDim)

		LoadMatrix(buf, r.blocks[i].attn.timeDecay.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].attn.timeFirst.w.Data, r.conf.embDim)

		LoadMatrix(buf, r.blocks[i].attn.timeMixK.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].attn.timeMixV.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].attn.timeMixR.w.Data, r.conf.embDim)

		LoadMatrix(buf, r.blocks[i].attn.key.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].attn.value.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].attn.receptance.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].attn.output.w.Data, r.conf.embDim)

		LoadMatrix(buf, r.blocks[i].ln2.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].ln2.b.Data, r.conf.embDim)

		LoadMatrix(buf, r.blocks[i].fnn.timeMixK.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].fnn.timeMixR.w.Data, r.conf.embDim)

		LoadMatrix(buf, r.blocks[i].fnn.key.w.Data, 3*r.conf.ctxLen)
		LoadMatrix(buf, r.blocks[i].fnn.receptance.w.Data, r.conf.embDim)
		LoadMatrix(buf, r.blocks[i].fnn.value.w.Data, r.conf.embDim)
	}
	LoadMatrix(buf, r.lnF.w.Data, r.conf.embDim)
	LoadMatrix(buf, r.lnF.b.Data, r.conf.embDim)

	LoadMatrix(buf, r.head.w.Data, r.conf.vocabSize)
}

func (r RWKV) Forward(token int, state [][]data.Tensor) (o data.Tensor, newState [][]data.Tensor) {
	x := r.emb.w.SubTensor([]int{token}).Transpose()
	x = utils.RWKVLayerNorm(x, r.ln0.w, r.ln0.b)
	var dx data.Tensor
	for i, block := range r.blocks {
		tmpX := utils.RWKVLayerNorm(x, block.ln1.w, block.ln1.b)
		dx, state[i][0], state[i][1], state[i][2] = timeMixing(tmpX, state[i][0], state[i][1], state[i][2], block.attn)
		x = x.Add(dx)

		tmpX = utils.RWKVLayerNorm(x, block.ln2.w, block.ln2.b)
		dx, state[i][3] = channelMixing(tmpX, state[i][3], block.fnn)
		x = x.Add(dx)
	}
	o = utils.RWKVLayerNorm(x, r.lnF.w, r.lnF.b)
	o = r.head.w.Dot(o)
	return o, state
}

func (r RWKV) Generate(tokens []int, maxLen int) (seq []int) {
	states := make([][]data.Tensor, 0)
	for i := 0; i < r.conf.nLayers; i++ {
		tmpState := make([]data.Tensor, 0)
		for j := 0; j < 4; j++ {
			tmp := data.NewTensor(nil, []int{r.conf.embDim, 1})
			tmpState = append(tmpState, tmp)
		}
		states = append(states, tmpState)
	}
	var o, probs data.Tensor
	var token int
	for _, token := range tokens {
		o, states = r.Forward(token, states)
	}

	for i := 0; i < maxLen; i++ {
		probs = utils.Softmax(o, 0).Transpose()
		token = SamplesProbs(probs.Data[0], 0.85, 5)
		seq = append(seq, token)
		o, states = r.Forward(token, states)
	}
	return seq
}
