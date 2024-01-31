#!/usr/bin/env python3
# -*- coding utf-8 -*-

from transformers import AutoTokenizer, AutoModel

'''
# 从Hugging Face下载模型
'''

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
    model = model.eval()

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
