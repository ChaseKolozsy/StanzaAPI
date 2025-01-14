# StanzaAPI

StanzaAPI is a high-performance, async API designed to interact with Stanza through FastAPI. This tool simplifies the process of utilizing Stanza's features via a Dockerized environment, optimized for concurrent processing.

## Features
- Asynchronous processing with FastAPI
- Automatic API documentation (Swagger UI)
- Concurrent request handling
- Pre-loaded language models
- Docker support with resource optimization

## Installation

### Local Installation
To install and run StanzaAPI locally, follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application by running:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5004
   ```

### Docker Installation
To run StanzaAPI using Docker, follow these steps:
1. Clone this repository to your local machine.
2. Build the Docker container using the provided Dockerfile:
   ```bash
   docker build -t stanza-service .
   ```
3. Run the Docker container:
   ```bash
   docker run -p 5004:5004 --name stanza-api stanza-service
   ```

## Usage

### API Documentation
Once running, you can access:
- Swagger UI documentation: `http://localhost:5004/docs`
- ReDoc documentation: `http://localhost:5004/redoc`

### API Endpoints

#### Process Text
```http
POST /process
Content-Type: application/json

{
    "language": "en",
    "text": "Your text to analyze"
}
```

Example response:
```json
[
    {
        "text": "Your text to analyze",
        "tokens": [
            {
                "text": "Your",
                "lemma": "your",
                "pos": "PRON",
                "deprel": "nmod"
            },
            ...
        ]
    }
]
```

## Performance Optimization
The API is optimized for concurrent processing and can handle multiple requests simultaneously. It uses:
- Thread pool for processing
- Pre-loaded language models
- Async request handling
- Optimized batch processing

## Docker Compose
For production deployments, you can use Docker Compose to run multiple instances with load balancing:
```bash
docker-compose up -d
```

## Requirements
- Python 3.11+
- FastAPI
- Uvicorn
- Stanza
- Pydantic

## License
[Your License Here]
