import os
import base64
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(
    title="Rafael AI Image API",
    version="1.0.0"
)

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

MODEL = "@cf/black-forest-labs/flux-1-schnell"


class ImageRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {
        "name": "Rafael AI Image API",
        "status": "online",
        "model": MODEL
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate")
async def generate(data: ImageRequest):

    if not ACCOUNT_ID or not API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="API não configurada no servidor"
        )

    if not data.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="O prompt não pode estar vazio"
        )

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{ACCOUNT_ID}/ai/run/{MODEL}"
    )

    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                url,
                headers=headers,
                json={
                    "prompt": data.prompt
                }
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Erro no serviço de geração"
            )

        content_type = response.headers.get(
            "content-type", ""
        )

        if "image/" in content_type:
            return Response(
                content=response.content,
                media_type=content_type
            )

        result = response.json()

        image_data = result.get("result", {}).get("image")

        if not image_data:
            raise HTTPException(
                status_code=502,
                detail="Resposta de imagem inválida"
            )

        return {
            "image_base64": image_data,
            "format": "base64"
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Tempo limite de geração excedido"
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Não foi possível conectar ao provedor"
        )
