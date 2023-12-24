# go LLM

Simple Implementation in GoLang to make inference of LLM models. No dependencies needed.

# How to run

Convert the weights of GPT2 to a binary file. You can use the following python script.

```sh
GPT2_MODEL_PATH=<PATH_TO_MODEL> GPT2_CONVERTED_MODEL_PATH=<PATH_TO_CONVERTED_MODEL> python res/converter.py
```

To start generating, run the following command.

```sh
GPT2_CONVERTED_MODEL_PATH=<PATH_TO_CONVERTED_MODEL> GPT2_TOKENIZER_PATH=<PATH_TO_TOKENIZER> go run main.go "hello my name is "
```
