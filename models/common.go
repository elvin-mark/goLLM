package models

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/elvin-mark/goLLM/data"
)

type Param struct {
	w data.Tensor
	b data.Tensor
}

type Pair struct {
	a string
	b string
}

func SamplesProbs(probs []float32, temperature float32, topK int) int {
	maxProb := float32(math.Inf(-1))
	maxIdx := 0
	for i, prob := range probs {
		if prob > maxProb {
			maxIdx = i
			maxProb = prob
		}
	}
	return maxIdx
}

func bytesToUnicode() (dict map[byte]string) {
	dict = make(map[byte]string)
	for c := '!'; c < '~'; c++ {
		dict[byte(c)] = string(c)
	}
	for c := '¡'; c < '¬'; c++ {
		dict[byte(c)] = string(c)
	}
	for c := '®'; c < 'ÿ'; c++ {
		dict[byte(c)] = string(c)
	}
	n := 0
	for i := 0; i < 256; i++ {
		if _, ok := dict[byte(i)]; !ok {
			dict[byte(i)] = string(rune(256 + n))
			n += 1
		}
	}
	return
}

func getPairs(word []string) (pairs []Pair) {
	prev := word[0]
	for _, curr := range word[1:] {
		pairs = append(pairs, Pair{
			a: prev,
			b: curr,
		})
		prev = curr
	}
	return
}

func index(word []string, pattern string) int {
	for i, w := range word {
		if w == pattern {
			return i
		}
	}
	return -1
}

type Encoder struct {
	pattern     *regexp.Regexp
	encoder     map[string]int
	decoder     map[int]string
	byteEncoder map[byte]string
	byteDecoder map[string]byte
	cache       map[string][]string
	bpeRanks    map[Pair]int
}

func NewEncoder(tokenizerPath string) (enc Encoder) {
	f, err := os.Open(tokenizerPath)
	if err != nil {
		log.Fatalf("could not load encoder: %v", err)
		return
	}
	bs, err := ioutil.ReadAll(f)
	tokenizer := make(map[string]any)
	if json.Unmarshal(bs, &tokenizer) != nil {
		return
	}

	model := tokenizer["model"].(map[string]any)
	tmp := model["vocab"].(map[string]any)
	encoder := make(map[string]int)
	decoder := make(map[int]string)
	for k, v := range tmp {
		idx := int(v.(float64))
		decoder[idx] = k
		encoder[k] = idx
	}

	byteEncoder := bytesToUnicode()
	byteDecoder := make(map[string]byte)
	for k, v := range byteEncoder {
		byteDecoder[v] = k
	}

	pattern := regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(\?!\S)|\s+`)

	merges := model["merges"].([]any)
	bpeRanks := make(map[Pair]int)
	for idx, w := range merges {
		elems := strings.Split(w.(string), " ")
		pair := Pair{
			a: elems[0],
			b: elems[1],
		}
		bpeRanks[pair] = idx
	}
	return Encoder{
		pattern:     pattern,
		encoder:     encoder,
		decoder:     decoder,
		byteEncoder: byteEncoder,
		byteDecoder: byteDecoder,
		cache:       make(map[string][]string),
		bpeRanks:    bpeRanks,
	}
}

func (e Encoder) GetRank(pair Pair) int {
	if val, ok := e.bpeRanks[pair]; ok {
		return val
	}
	return 100000000
}

func (e Encoder) MinPair(pairs []Pair) (p Pair) {
	p = pairs[0]
	for _, pair := range pairs[1:] {
		if e.GetRank(pair) < e.GetRank(p) {
			p = pair
		}
	}
	return
}

func (e Encoder) BPE(token string) (elems []string) {
	if val, ok := e.cache[token]; ok {
		return val
	}
	word := make([]string, 0)
	for _, c := range token {
		word = append(word, string(c))
	}
	pairs := getPairs(word)
	if len(pairs) == 0 {
		return word
	}

	for {
		pair := e.MinPair(pairs)
		if _, ok := e.bpeRanks[pair]; !ok {
			break
		}
		first := pair.a
		second := pair.b
		newWord := make([]string, 0)
		for i := 0; i < len(word); {
			j := index(word[i:], first)
			if j >= 0 {
				j += i
				newWord = append(newWord, word[i:j]...)
				i = j
			} else {
				newWord = append(newWord, word[i:]...)
				break
			}
			if word[i] == first && i < len(word)-1 && word[i+1] == second {
				newWord = append(newWord, first+second)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i += 1
			}
		}
		word = newWord
		if len(word) == 1 {
			break
		} else {
			pairs = getPairs(word)
		}
	}
	e.cache[token] = word
	return word
}

func (e Encoder) Encode(sentence string) (tokens []int) {
	for _, word := range e.pattern.FindAllString(sentence, -1) {
		proc := make([]string, 0)
		for _, b := range word {
			buf := make([]byte, utf8.UTFMax)
			n := utf8.EncodeRune(buf, b)
			for i := 0; i < n; i++ {
				proc = append(proc, e.byteEncoder[buf[i]])
			}
		}

		for _, elem := range e.BPE(strings.Join(proc, "")) {
			tokens = append(tokens, e.encoder[elem])
		}
	}
	return
}

func (e Encoder) Decode(tokens []int) (sentence string) {
	res := make([]string, 0)
	for _, token := range tokens {
		res = append(res, e.decoder[token])
	}
	buf := make([]byte, 0)
	for _, b := range strings.Join(res, "") {
		buf = append(buf, e.byteDecoder[string(b)])
	}
	tmp := make([]string, 0)
	for i := 0; i < len(buf); {
		r, n := utf8.DecodeRune(buf[i:])
		tmp = append(tmp, string(r))
		i += n
	}
	return strings.Join(tmp, "")
}
