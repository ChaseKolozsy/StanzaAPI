from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import stanza
import hashlib
import time
from collections import OrderedDict

app = FastAPI(title="Stanza API", version="1.0.0")

device = "gpu"
cuda = 1
stanza_pool = None

# LRU Cache for results
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[List]:
        if key in self.cache:
            # Move to end to show recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: List) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

# Initialize cache
results_cache = LRUCache()

def generate_cache_key(texts: List[str]) -> str:
    """Generate a unique key for the batch of texts"""
    combined = "".join(texts)
    return hashlib.md5(combined.encode()).hexdigest()

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
    def __init__(self, device, cuda):
        self.pipeline = None
        self.batch_size = 4096
        self.device = device
        self.cuda = cuda
        self.use_gpu = 'gpu' == self.device

    async def initialize(self, language: str):
        """Initialize a single pipeline for the specified language"""
        self.pipeline = stanza.Pipeline(
            lang=language,
            processors='tokenize,pos,lemma,depparse',
            use_gpu=self.use_gpu,
            device=self.device if not self.use_gpu else f'cuda:{self.cuda}',
            batch_size=self.batch_size,
            preload_processors=True
        )


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
        
        # Generate cache key for this batch
        cache_key = generate_cache_key(request.texts)
        
        # Check cache first
        cached_results = results_cache.get(cache_key)
        #if cached_results is not None:
        #    return cached_results
        
        # Process new request
        results = process_with_stanza(stanza_pool.pipeline, request.texts)
        
        # Cache results before returning
        results_cache.put(cache_key, results)
        
        return results
    except Exception as e:
        # Check cache again in case of error
        # This handles the case where processing succeeded but response failed
        cached_results = results_cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stanza API Server')
    parser.add_argument('--port', type=int, default=5006,
                      help='Port to run the server on (default: 5004)')
    parser.add_argument('--device', type=str, default="gpu", help="cpu or gpu")
    parser.add_argument('--cuda', type=int, default="1", help="choose which gpu you want, 0, 1, etc.")
    
    args = parser.parse_args()
    cuda = args.cuda
    device = args.device
    print(f"args: {args}")
    stanza_pool = StanzaPool(device=device, cuda=cuda)
    print("device: {stanza_pool.device}, cuda: {stanza_pool.cuda}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        workers=1,
        loop="auto",
        reload=False
    )
