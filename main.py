from dotenv import load_dotenv
load_dotenv(override = True)

from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from service import *
from schema import *

app = FastAPI()

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    completion = process_request(request)

    if request.stream:        
        return StreamingResponse(
            chat_stream_to_sglang_response(completion),
            media_type = "text/event-stream"
        )
    else:
        return completion

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host = "0.0.0.0", port = 4000, reload = True)
