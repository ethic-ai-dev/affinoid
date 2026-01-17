from dotenv import load_dotenv
load_dotenv(override = True)

from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from schema import *

import json
import time
import uuid

app = FastAPI()

def stream_response(user_message: str):
    response_text = f"Echo streaming: {user_message}"

    for char in response_text:
        chunk = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {"content": char},
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(0.05)

    done_chunk = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion.chunk",
        "choices": [
            {
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_message = request.messages[-1].content if request.messages else ""

    if request.stream:
        return StreamingResponse(
            stream_response(user_message),
            media_type="text/event-stream"
        )

    assistant_response = f"Echo: {user_message}"
    response = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_response},
                "finish_reason": "stop"
            }
        ]
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host = "0.0.0.0", port = 4000, reload = True)
