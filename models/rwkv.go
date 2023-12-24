package models

import (
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
	nLayers int
	emdDim  int
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

func NewRWKV(conf RWKVConfig) RWKV {
	return RWKV{
		conf: conf,
	}
}

func (r RWKV) Init() {

}

func (r RWKV) Forward(token int, state [][]data.Tensor) (o data.Tensor, newState [][]data.Tensor) {
	x := r.emb.w.SubTensor([]int{token})
	x = utils.LayerNorm(x, r.ln0.w, r.ln0.b)
	var dx data.Tensor
	for i, block := range r.blocks {
		tmpX := utils.LayerNorm(x, block.ln1.w, block.ln1.b)
		dx, state[i][0], state[i][1], state[i][2] = timeMixing(tmpX, state[i][0], state[i][1], state[i][2], block.attn)
		x = x.Add(dx)

		tmpX = utils.LayerNorm(x, block.ln2.w, block.ln2.b)
		dx, state[i][3] = channelMixing(tmpX, state[i][3], block.fnn)
		x = x.Add(dx)
	}
	o = utils.LayerNorm(x, r.lnF.w, r.lnF.b)
	return o, state
}

func (r RWKV) Generate(tokens []int, maxLen int) (seq []int) {
	states := make([][]data.Tensor, r.conf.nLayers)
	for i := 0; i < r.conf.nLayers; i++ {
		tmpState := make([]data.Tensor, 4)
		for j := 0; j < 4; j++ {
			tmp := data.NewTensor(nil, []int{1, r.conf.emdDim})
			tmpState = append(tmpState, tmp)
		}
		states = append(states, tmpState)
	}
	headWT := r.head.w.Transpose()
	var o, probs data.Tensor
	var token int
	for _, token := range tokens {
		o, states = r.Forward(token, states)
	}

	for i := 0; i < maxLen; i++ {
		o = o.Dot(headWT)
		probs = utils.Softmax(o, 1)
		token = SamplesProbs(probs.Data[0], 0.85, 5)
		seq = append(seq, token)
		o, states = r.Forward(token, states)
	}
	return seq
}
