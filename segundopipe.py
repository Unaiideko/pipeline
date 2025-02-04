title: Mistral Index Ollama Pipeline
author: open-webui
date: 2025-02-04
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using Mistral model from Ollama.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama


from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "mistral:7b"),  # Usar el modelo Mistral
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),  # Puedes cambiarlo según el embedding específico que uses.
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,  # Usar el modelo Mistral aquí
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # Esta función se llama cuando el servidor se inicia.
        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

    async def on_shutdown(self):
        # Esta función se llama cuando el servidor se detiene.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # Aquí puedes agregar tu pipeline personalizado de RAG.
        # Recuperar información relevante de tu base de conocimiento y sintetizarla para generar una respuesta.
        
        print(f"Mensaje recibido: {user_message}")
        print(f"Mensajes previos: {messages}")
        
        # Inicializa el motor de consulta usando el índice previamente construido
        query_engine = self.index.as_query_engine(streaming=True)
        
        # Consultar el motor con el mensaje del usuario
        response = query_engine.query(user_message)

        # Puede que quieras hacer streaming de la respuesta directamente o agregarla
        # Si haces streaming:
        for chunk in response.response_gen:
            yield chunk  # Streaming de los trozos de respuesta a medida que se procesan.
        
        # O puedes devolver la respuesta completa una vez que esté completamente procesada:
        # return response.response
