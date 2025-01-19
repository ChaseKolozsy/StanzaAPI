from fastapi import FastAPI, HTTPException
import torch
import uvicorn

app = FastAPI(title="Memory Cleaner API", version="1.0.0")

@app.post("/clear_memory")
async def clear_memory():
    try:
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Collect GPU memory cached by IPC
            torch.cuda.ipc_collect()
            return {"status": "success", "message": "GPU memory cleared"}
        else:
            return {"status": "skipped", "message": "No GPU available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory Cleaner API Server')
    parser.add_argument('--port', type=int, default=5050,
                      help='Port to run the server on (default: 5050)')
    
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        workers=1,
        loop="auto",
        reload=False
    )