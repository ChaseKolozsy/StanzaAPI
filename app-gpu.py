from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import stanza
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import asyncio
import platform
import torch

# Calculate optimal thread count for M3 Max
CPU_CORES = multiprocessing.cpu_count()
WORKER_COUNT = max(int((CPU_CORES - 2) / 2), 1)

app = FastAPI(title="Stanza API", version="1.0.0")
thread_pool = ThreadPoolExecutor(max_workers=WORKER_COUNT)

# Define data models
class TextRequest(BaseModel):
    language: str
    text: str

class Token(BaseModel):
    text: str
    lemma: str
    pos: str
    deprel: str

class Sentence(BaseModel):
    text: str
    tokens: List[Token]

class StanzaPool:
    def __init__(self):
        self.pipeline = None
        self.batch_size = 4096
        
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print("🚀 CUDA GPU acceleration enabled!")
        else:
            print("⚙️ Using CPU processing")

    async def initialize(self, language: str):
        """Initialize a single pipeline for the specified language"""
        self.pipeline = stanza.Pipeline(
            lang=language,
            processors='tokenize,pos,lemma,depparse',
            use_gpu=self.use_gpu,
            batch_size=self.batch_size,
            preload_processors=True
        )

stanza_pool = StanzaPool()

def process_with_stanza(pipeline: stanza.Pipeline, texts: List[str]) -> List[List[Sentence]]:
    # Join texts with double newlines for Stanza's batch processing
    batch_text = "\n\n".join(texts)
    doc = pipeline(batch_text)
    
    # Track which sentences belong to which original text
    result = []
    current_sentences = []
    sent_count = 0
    texts_processed = 0
    
    for sent in doc.sentences:
        current_sentences.append(Sentence(
            text=sent.text,
            tokens=[
                Token(
                    text=word.text,
                    lemma=word.lemma,
                    pos=word.pos,
                    deprel=word.deprel
                ) for word in sent.words
            ]
        ))
        sent_count += 1
        
        # If we've found a blank line or reached the end, start a new document
        if sent.text.strip() == "" or sent_count == len(doc.sentences):
            if current_sentences:
                result.append([s for s in current_sentences if s.text.strip()])
                current_sentences = []
                texts_processed += 1
    
    return result

@app.on_event("startup")
async def startup_event():
    # Initialize multiple pipelines for English only
    await stanza_pool.initialize('hu')  # or whatever language you want

@app.post("/process", response_model=List[Sentence])
async def process_text(request: TextRequest):
    try:
        if not stanza_pool.pipeline:
            raise HTTPException(status_code=400, detail="Language model not initialized")
        
        result = process_with_stanza(stanza_pool.pipeline, [request.text])
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add new model for batch requests
class BatchTextRequest(BaseModel):
    language: str
    texts: List[str]

# Add new batch processing endpoint
@app.post("/batch_process", response_model=List[List[Sentence]])
async def batch_process_texts(request: BatchTextRequest):
    try:
        if not stanza_pool.pipeline:
            raise HTTPException(status_code=400, detail="Language model not initialized")
        
        results = process_with_stanza(stanza_pool.pipeline, request.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stanza API Server')
    parser.add_argument('--port', type=int, default=5004,
                      help='Port to run the server on (default: 5004)')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        workers=1,  # Using thread pool instead of multiple workers
        loop="auto"
    )
