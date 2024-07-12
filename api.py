from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Tuple, Dict
import json
from gradio_client import Client
import asyncio
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gradio client
gradio_url = os.getenv("GRADIO_URL", "http://127.0.0.1:7860/")
gradio_client = Client(gradio_url)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Get the last user message
        last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")

        # Prepare the chat history
        history = []
        for msg in request.messages:
            if msg.role == "user":
                history.append([msg.content, None])
            elif msg.role == "assistant" and history:
                history[-1][1] = msg.content

        result = await asyncio.to_thread(
            gradio_client.predict,
            last_user_message,
            history,
            api_name="/chat"
        )
        
        # Extracting the response from the Gradio result
        chat_history, processing_log = result
        response = chat_history[-1][1] if chat_history else ""

        # Construct the response
        choice = Choice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )

        # Dummy usage data (you might want to implement actual token counting)
        usage = Usage(prompt_tokens=len(last_user_message), completion_tokens=len(response), total_tokens=len(last_user_message)+len(response))

        return ChatCompletionResponse(
            id="chatcmpl-" + os.urandom(4).hex(),
            object="chat.completion",
            created=int(asyncio.get_event_loop().time()),
            model=request.model,
            choices=[choice],
            usage=usage
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return {
        "data": [
            {
                "id": "moa",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner"
            }
        ],
        "object": "list"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        ssl_keyfile=os.getenv("SSL_KEYFILE", None),
        ssl_certfile=os.getenv("SSL_CERTFILE", None),
    )