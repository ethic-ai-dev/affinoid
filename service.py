from openai import OpenAI
from schema import *

import json
import os

__is_gpu_working__ = False

def check_if_affine_request(request: ChatCompletionRequest):
    if request.stream: return False
    return True

def call_deepseek_api(request: ChatCompletionRequest):
    request.model = "deepseek-reasoner"
    return __call_llm_api__(request, "https://api.deepseek.com", os.environ["DEEPSEEK_KEY"])

def call_openai_api(request: ChatCompletionRequest, model: str):
    request.model = model
    return __call_llm_api__(request, "https://api.openai.com/v1", os.environ["OPENAI_KEY"])

def __call_llm_api__(request: ChatCompletionRequest, api_url: str, api_key: str):
    client = OpenAI(api_key = api_key, base_url = api_url)
    messages = request.model_dump()["messages"]

    response = client.chat.completions.create(
        model = request.model,
        messages = messages,
        stream = request.stream,
        max_completion_tokens = request.max_completion_tokens
    )    
    return response

def chat_stream_to_sglang_response(stream):
    for chunk in stream:
        if not chunk.choices: continue
        delta = chunk.choices[0].delta

        if not delta or (not getattr(delta, "content", None) and not getattr(delta, "reasoning_content", None)): continue

        sg_chunk = {
            "id": chunk.id,
            "object": "chat.completion.chunk",
            "created": chunk.created,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": getattr(delta, "reasoning_content", None),
                        "content": getattr(delta, "content", None)
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(sg_chunk)}\n\n"

    yield "data: [DONE]\n\n"

def chat_block_to_text(block):
    msg = block.choices[0].message

    reasoning_content = getattr(msg, "reasoning_content", None)
    content = getattr(msg, "content", None)

    return reasoning_content, content
