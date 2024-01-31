#!/usr/bin/env python3
# -*- coding utf-8 -*-

from modelscope import AutoTokenizer, AutoModel, snapshot_download

'''
# 从modelscope下载模型
'''

if __name__ == "__main__":
    model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    model = model.eval()

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
