#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
import platform
from transformers import AutoTokenizer, AutoModel

'''
循环聊天
'''

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

bad_words = ["你好", "ChatGLM"]
bad_word_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)

    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break

        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue

        print("\nChatGLM：", end="")
        current_length = 0
        response_generated = False
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            response_generated = True
            if stop_stream:
                stop_stream = False
                break
            else:
                # Check if the response contains any bad words
                if any(bad_word in response[current_length:] for bad_word in bad_words):
                    print("=== bad words ===")
                else:
                    print(response[current_length:], end="", flush=True)
                    current_length = len(response)

        if not response_generated:
                print("没有生成任何回答。")

        print("")


if __name__ == "__main__":
    main()
