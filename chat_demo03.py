#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
from typing import Union, Annotated
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TextIteratorStreamer
)

'''
聊天，按流处理
'''

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

def load_model_and_tokenizer() :
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
    return model, tokenizer

def predict(history, max_length, top_p, temperature):
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.2,
    }

    model.generate(generate_kwargs)

    for new_token in streamer:
        if new_token != '':
            history[-1][1] += new_token
            yield history


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)
    question =  (('按步骤说明，如何做西红柿炒蛋','1.西红柿洗净，切成小块；鸡蛋打散备用。2.热锅凉油，将鸡蛋液煎至金黄色捞出沥干备用。3.将锅中余油留少许，放入葱姜蒜末爆香。4.加入西红柿翻炒至断生。5.加入少许盐、白糖调味，继续翻炒至快出汁时加入鸡蛋，快速搅拌均匀即可起锅。'),('按步骤说明，如何做红烧肉',''))
    answer = predict(question,8192,0.8,0.6)
    print(answer[-1][1])
