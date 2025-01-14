from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import stanza
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import asyncio

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
    def __init__(self, num_pipelines: int = WORKER_COUNT):
        self.num_pipelines = num_pipelines
        self.pipelines: List[stanza.Pipeline] = []
        self.locks: List[asyncio.Lock] = []
        self.batch_size = 4096
        self.current_pipeline = 0

    def get_pipeline(self):
        """Get the next available pipeline in a round-robin fashion"""
        pipeline = self.pipelines[self.current_pipeline]
        lock = self.locks[self.current_pipeline]
        self.current_pipeline = (self.current_pipeline + 1) % self.num_pipelines
        return pipeline, lock

    async def initialize(self, language: str):
        """Initialize multiple pipelines for the specified language"""
        self.pipelines = [
            stanza.Pipeline(
                lang=language,
                processors='tokenize,pos,lemma,depparse',
                use_gpu=False,
                batch_size=self.batch_size,
                preload_processors=True
            ) for _ in range(self.num_pipelines)
        ]
        self.locks = [asyncio.Lock() for _ in range(self.num_pipelines)]

stanza_pool = StanzaPool()

def process_with_stanza(pipeline: stanza.Pipeline, text: str) -> List[Sentence]:
    doc = pipeline(text)
    result = []
    for sent in doc.sentences:
        sentence = Sentence(
            text=sent.text,
            tokens=[
                Token(
                    text=word.text,
                    lemma=word.lemma,
                    pos=word.pos,
                    deprel=word.deprel
                ) for word in sent.words
            ]
        )
        result.append(sentence)
    return result

@app.on_event("startup")
async def startup_event():
    # Initialize multiple pipelines for English only
    await stanza_pool.initialize('hu')  # or whatever language you want

@app.post("/process", response_model=List[Sentence])
async def process_text(request: TextRequest):
    try:
        if not stanza_pool.pipelines:
            raise HTTPException(status_code=400, detail="Language model not initialized")
        
        # Get next available pipeline and lock using the pool's method
        pipeline, lock = stanza_pool.get_pipeline()
        
        async with lock:
            result = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                process_with_stanza,
                pipeline,
                request.text
            )
        
        return result
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
        if not stanza_pool.pipelines:
            raise HTTPException(status_code=400, detail="Language model not initialized")
        
        # Split texts into chunks based on number of pipelines
        num_pipelines = len(stanza_pool.pipelines)
        chunk_size = (len(request.texts) + num_pipelines - 1) // num_pipelines
        text_chunks = [request.texts[i:i + chunk_size] 
                      for i in range(0, len(request.texts), chunk_size)]
        
        async def process_chunk(chunk, pipeline_idx):
            pipeline = stanza_pool.pipelines[pipeline_idx]
            lock = stanza_pool.locks[pipeline_idx]
            async with lock:
                return await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    lambda: [process_with_stanza(pipeline, text) for text in chunk]
                )
        
        # Process chunks concurrently using different pipelines
        tasks = [
            process_chunk(chunk, idx) 
            for idx, chunk in enumerate(text_chunks)
        ]
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results back into a single list
        results = []
        for chunk in chunk_results:
            results.extend(chunk)
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5004,
        workers=1,  # Using thread pool instead of multiple workers
        loop="auto"
    )