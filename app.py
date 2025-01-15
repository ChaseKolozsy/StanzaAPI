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
        if not stanza_pool.pipelines:
            raise HTTPException(status_code=400, detail="Language model not initialized")
        
        pipeline, lock = stanza_pool.get_pipeline()
        
        async with lock:
            result = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                process_with_stanza,
                pipeline,
                [request.text]  # Pass as single-item list
            )
        
        return result[0]  # Return first (and only) item
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
        
        # Calculate optimal batch size (adjust these numbers based on your needs)
        MAX_TEXTS_PER_BATCH = 50
        MAX_CHARS_PER_BATCH = 100000
        
        def create_batches(texts):
            current_batch = []
            current_chars = 0
            batches = []
            
            for text in texts:
                if (len(current_batch) >= MAX_TEXTS_PER_BATCH or 
                    current_chars + len(text) > MAX_CHARS_PER_BATCH):
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0
                current_batch.append(text)
                current_chars += len(text)
            
            if current_batch:
                batches.append(current_batch)
            return batches
        
        # Split texts into reasonable sized batches
        text_batches = create_batches(request.texts)
        
        async def process_batch(batch, pipeline_idx):
            pipeline = stanza_pool.pipelines[pipeline_idx]
            lock = stanza_pool.locks[pipeline_idx]
            async with lock:
                return await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    process_with_stanza,
                    pipeline,
                    batch
                )
        
        # Process batches concurrently using different pipelines
        tasks = []
        for idx, batch in enumerate(text_batches):
            pipeline_idx = idx % len(stanza_pool.pipelines)
            tasks.append(process_batch(batch, pipeline_idx))
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch in batch_results:
            results.extend(batch)
            
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