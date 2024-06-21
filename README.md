# StanzaAPI

StanzaAPI is a user-friendly tool designed to interact with Stanza through a Docker API. This tool simplifies the process of utilizing Stanza's features via a Dockerized environment.

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
   python app.py
   ```

### Docker Installation
To run StanzaAPI using Docker, follow these steps:
1. Clone this repository to your local machine.
2. Build the Docker container using the provided Dockerfile:
   ```bash
   docker build -t stanzaapi .
   ```
3. Run the Docker container:
   ```bash
   docker run -p 5004:5004 stanzaapi
   ```

## Usage
Once you have installed StanzaAPI, you can access the API at `http://localhost:5004/`.
