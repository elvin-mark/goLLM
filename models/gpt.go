package models

import (
	"bytes"
	"encoding/binary"
	"log"
	"os"

	"github.com/elvin-mark/goLLM/data"
	"github.com/elvin-mark/goLLM/utils"
)

type MLP struct {
	fc   Param
	proj Param
}

type GPT2Attention struct {
	attn Param
	proj Param
}

type GPT2Block struct {
	ln1  Param
	attn GPT2Attention
	ln2  Param
	mlp  MLP
}

type GPT2Config struct {
	nHeads    int
	nLayers   int
	embDim    int
	ctxLen    int
	vocabSize int
}

type GPT2 struct {
	conf   GPT2Config
	wpe    Param
	wte    Param
	blocks []GPT2Block
	lnF    Param
}

func attention(q data.Tensor, k data.Tensor, v data.Tensor, mask data.Tensor) (r data.Tensor) {
	o := q.Dot(k.Transpose())
	tmp := data.NewTensor([][]float32{{data.Sqrt(float32(q.Shape[1]))}}, []int{1, 1})
	r = utils.Softmax(o.Div(tmp).Add(mask), 1).Dot(v)
	return
}

func ffn(x data.Tensor, mlp MLP) (r data.Tensor) {
	o := utils.Linear(x, mlp.fc.w, mlp.fc.b)
	o = o.Apply(utils.GeLU)
	r = utils.Linear(o, mlp.proj.w, mlp.proj.b)
	return r
}

func splitQKVTensor(x data.Tensor, nHeads int) (qs, ks, vs []data.Tensor) {
	rows := x.Shape[0]
	cols := x.Shape[1]
	newCols := cols / (3 * nHeads)

	for h := 0; h < nHeads; h++ {
		q := data.NewTensor(nil, []int{rows, newCols})
		k := data.NewTensor(nil, []int{rows, newCols})
		v := data.NewTensor(nil, []int{rows, newCols})

		for i := 0; i < rows; i++ {
			q.Data[i] = x.Data[i][newCols*h : (h+1)*newCols]
			k.Data[i] = x.Data[i][nHeads*newCols+newCols*h : nHeads*newCols+(h+1)*newCols]
			v.Data[i] = x.Data[i][nHeads*2*newCols+newCols*h : nHeads*2*newCols+(h+1)*newCols]
		}

		qs = append(qs, q)
		ks = append(ks, k)
		vs = append(vs, v)
	}
	return
}
func hstackTensor(ts []data.Tensor) (r data.Tensor) {
	rows := ts[0].Shape[0]
	cols := ts[0].Shape[1]
	N := len(ts)
	r = data.NewTensor(nil, []int{rows, N * cols})
	for i := 0; i < rows; i++ {
		for k := 0; k < N; k++ {
			for j := 0; j < cols; j++ {
				r.Data[i][k*cols+j] = ts[k].Data[i][j]
			}
		}
	}
	return
}

func mha(x data.Tensor, attn GPT2Attention, conf GPT2Config) (r data.Tensor) {
	o := utils.Linear(x, attn.attn.w, attn.attn.b)
	qs, ks, vs := splitQKVTensor(o, conf.nHeads)
	tmp := make([]data.Tensor, 0)
	mask := data.NewTensor(nil, []int{x.Shape[0], x.Shape[0]})
	mask.UpTri()
	fn := func(x float32) float32 {
		return x * -10000000000
	}
	mask = mask.Apply(fn)
	for i := 0; i < conf.nHeads; i++ {
		tmp = append(tmp, attention(qs[i], ks[i], vs[i], mask))
	}
	o = hstackTensor(tmp)
	r = utils.Linear(o, attn.proj.w, attn.proj.b)
	return
}

func transformerBlock(x data.Tensor, block GPT2Block, conf GPT2Config) data.Tensor {
	o := utils.LayerNorm(x, block.ln1.w, block.ln1.b)
	o = mha(o, block.attn, conf)
	o = o.Add(x)

	r := utils.LayerNorm(o, block.ln2.w, block.ln2.b)
	r = ffn(r, block.mlp)
	r = r.Add(o)
	return r
}

func (g *GPT2) Init() {
	{
		tmpW := data.NewTensor(nil, []int{g.conf.vocabSize, g.conf.embDim})
		tmpW.Random()
		g.wte = Param{
			w: tmpW,
		}
	}
	{
		tmpW := data.NewTensor(nil, []int{g.conf.ctxLen, g.conf.embDim})
		tmpW.Random()
		g.wpe = Param{
			w: tmpW,
		}
	}
	g.blocks = make([]GPT2Block, g.conf.nLayers)
	for i := range g.blocks {
		tmpW := data.NewTensor(nil, []int{1, g.conf.embDim})
		tmpW.Random()
		tmpb := data.NewTensor(nil, []int{1, g.conf.embDim})
		tmpb.Random()
		ln1 := Param{
			w: tmpW,
			b: tmpb,
		}
		tmpW = data.NewTensor(nil, []int{g.conf.embDim, 3 * g.conf.embDim})
		tmpW.Random()
		tmpb = data.NewTensor(nil, []int{1, 3 * g.conf.embDim})
		tmpb.Random()
		attn := Param{
			w: tmpW,
			b: tmpb,
		}
		tmpW = data.NewTensor(nil, []int{g.conf.embDim, g.conf.embDim})
		tmpW.Random()
		tmpb = data.NewTensor(nil, []int{1, g.conf.embDim})
		tmpb.Random()
		proj := Param{
			w: tmpW,
			b: tmpb,
		}
		attnLayer := GPT2Attention{
			attn: attn,
			proj: proj,
		}
		tmpW = data.NewTensor(nil, []int{1, g.conf.embDim})
		tmpW.Random()
		tmpb = data.NewTensor(nil, []int{1, g.conf.embDim})
		tmpb.Random()
		ln2 := Param{
			w: tmpW,
			b: tmpb,
		}
		tmpW = data.NewTensor(nil, []int{g.conf.embDim, 3 * g.conf.ctxLen})
		tmpW.Random()
		tmpb = data.NewTensor(nil, []int{1, 3 * g.conf.ctxLen})
		tmpb.Random()
		fc := Param{
			w: tmpW,
			b: tmpb,
		}
		tmpW = data.NewTensor(nil, []int{3 * g.conf.ctxLen, g.conf.embDim})
		tmpW.Random()
		tmpb = data.NewTensor(nil, []int{1, g.conf.embDim})
		tmpb.Random()
		proj = Param{
			w: tmpW,
			b: tmpb,
		}
		mlpLayer := MLP{
			fc:   fc,
			proj: proj,
		}
		block := GPT2Block{
			ln1:  ln1,
			attn: attnLayer,
			ln2:  ln2,
			mlp:  mlpLayer,
		}
		g.blocks[i] = block
	}

	tmpW := data.NewTensor(nil, []int{1, g.conf.embDim})
	tmpW.Random()
	tmpb := data.NewTensor(nil, []int{1, g.conf.embDim})
	tmpb.Random()
	g.lnF = Param{
		w: tmpW,
		b: tmpb,
	}
}

func (g GPT2) Forward(tokens []int) data.Tensor {
	l := make([]int, len(tokens))
	for i := range tokens {
		l[i] = i
	}
	te := g.wte.w.SubTensor(tokens)
	pe := g.wpe.w.SubTensor(l)
	x := te.Add(pe)
	for _, block := range g.blocks {
		x = transformerBlock(x, block, g.conf)
	}
	x = utils.LayerNorm(x, g.lnF.w, g.lnF.b)
	return x
}

func (g GPT2) Generate(tokens []int, maxLen int) (seq []int) {
	embWT := g.wte.w.Transpose()
	var newToken int
	var logits data.Tensor
	for i := 0; i < maxLen; i++ {
		o := g.Forward(tokens)
		logits = o.Dot(embWT)
		newToken = SamplesProbs(logits.Data[len(tokens)-1], 0.85, 5)
		tokens = append(tokens, newToken)
	}
	return tokens
}

func loadMatrix(buf *bytes.Reader, mat [][]float32, rows int) {
	for i := 0; i < rows; i++ {
		err := binary.Read(buf, binary.LittleEndian, mat[i])
		if err != nil {
			log.Fatalf("could not get matrix values: %v", err)
			return
		}
	}
}

func (g GPT2) LoadWeights(modelPath string) {
	bs, err := os.ReadFile(modelPath)
	if err != nil {
		return
	}
	buf := bytes.NewReader(bs)
	loadMatrix(buf, g.wte.w.Data, g.conf.vocabSize)
	loadMatrix(buf, g.wpe.w.Data, g.conf.ctxLen)

	for l := 0; l < g.conf.nLayers; l++ {
		loadMatrix(buf, g.blocks[l].ln1.w.Data, 1)
		loadMatrix(buf, g.blocks[l].ln1.b.Data, 1)

		loadMatrix(buf, g.blocks[l].attn.attn.w.Data, g.conf.embDim)
		loadMatrix(buf, g.blocks[l].attn.attn.b.Data, 1)

		loadMatrix(buf, g.blocks[l].ln2.w.Data, 1)
		loadMatrix(buf, g.blocks[l].ln2.b.Data, 1)

		loadMatrix(buf, g.blocks[l].attn.proj.w.Data, g.conf.embDim)
		loadMatrix(buf, g.blocks[l].attn.proj.b.Data, 1)

		loadMatrix(buf, g.blocks[l].mlp.fc.w.Data, g.conf.embDim)
		loadMatrix(buf, g.blocks[l].mlp.fc.b.Data, 1)

		loadMatrix(buf, g.blocks[l].mlp.proj.w.Data, 3*g.conf.ctxLen)
		loadMatrix(buf, g.blocks[l].mlp.proj.b.Data, 1)

	}
	loadMatrix(buf, g.lnF.w.Data, 1)
	loadMatrix(buf, g.lnF.b.Data, 1)
}

func NewGPT2Config() GPT2Config {
	return GPT2Config{
		nHeads:    12,
		nLayers:   12,
		ctxLen:    1024,
		embDim:    768,
		vocabSize: 50257,
	}
}
func NewGPT2(conf GPT2Config) GPT2 {
	return GPT2{
		conf: conf,
	}
}
