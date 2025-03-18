from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import os

# Poner aquí tu token de: https://huggingface.co/settings/tokens
HF_TOKEN = "<PUT_HERE_YOUR_TOKEN>" 

app = FastAPI()

# Crear el modelo de Pydantic para recibir el cuerpo de la solicitud
class PromptRequest(BaseModel):
    prompt: str

# Cargar el tokenizador y el modelo de Hugging Face con el token de autenticación
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=HF_TOKEN)

# Crear el pipeline para la generación de texto
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

@app.post("/generate")
async def generate_text(request: PromptRequest):
    # Configurar la generación de texto, añadiendo truncamiento explícito y configurando pad_token_id
    result = pipe(request.prompt, 
                  max_length=100, 
                  truncation=True,  # Aseguramos truncamiento explícito
                  pad_token_id=tokenizer.eos_token_id)  # Establecemos el pad_token_id

    return {"response": result[0]['generated_text']}
