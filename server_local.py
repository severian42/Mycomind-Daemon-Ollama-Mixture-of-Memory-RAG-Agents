import socket as s
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
from utils import generate_together, generate_with_references
from omoa import OllamaMixtureOfAgents, OllamaAgent
from duck_duck_go_websearch_agent import search_web

# Load environment variables
load_dotenv()

# Constants
HEADER = 64
FORMAT = "utf-8"
DISCONNECT_MSG = "DISCONNECT"

# Ollama configuration
OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", 4))
OLLAMA_MAX_LOADED_MODELS = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", 4))

# Model configuration
MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

def check_service_running(host, port):
    try:
        with s.create_connection((host, port), timeout=2):
            return True
    except (s.timeout, ConnectionRefusedError, OSError):
        print("Ollama not running")
        return False

if not check_service_running('localhost', 11434):
    exit()

def get_network_ip():
    so = s.socket(s.AF_INET, s.SOCK_DGRAM)
    so.connect(("8.8.8.8", 80))
    ip_address = so.getsockname()[0]
    so.close()
    return ip_address

async def generate_reference_response(executor, model, message):
    return await asyncio.get_event_loop().run_in_executor(
        executor, generate_together, model, [{"role": "user", "content": message}]
    )

async def process_message(mixture: OllamaMixtureOfAgents, msg: str):
    response, web_search_performed = await mixture.get_response(msg)
    return response, web_search_performed

async def handle_client(mixture: OllamaMixtureOfAgents, reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[NEW CONNECTION] {addr} connected")
    try:
        while True:
            msg_len = await reader.read(HEADER)
            if not msg_len:
                break
            msg_len = int(msg_len.decode(FORMAT))
            msg = await reader.read(msg_len)
            msg = msg.decode(FORMAT)
            
            if msg == DISCONNECT_MSG:
                break
            
            print(f'{addr} {msg}')
            response, web_search_performed = await process_message(mixture, msg)
            writer.write(response.encode(FORMAT))
            if web_search_performed:
                writer.write(b"\n[Web search was performed]")
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[DISCONNECTED] {addr} disconnected")

async def start_server(host, port):
    reference_agents = [
        OllamaAgent(MODEL_REFERENCE_1, "AnalyticalAgent", DEFAULT_PROMPTS["AnalyticalAgent"]),
        OllamaAgent(MODEL_REFERENCE_2, "HistoricalContextAgent", DEFAULT_PROMPTS["HistoricalContextAgent"]),
        OllamaAgent(MODEL_REFERENCE_3, "ScienceTruthAgent", DEFAULT_PROMPTS["ScienceTruthAgent"])
    ]
    final_agent = OllamaAgent(MODEL_AGGREGATE, "SynthesisAgent", DEFAULT_PROMPTS["SynthesisAgent"])

    mixture = OllamaMixtureOfAgents(reference_agents, final_agent)

    server = await asyncio.start_server(
        lambda r, w: handle_client(mixture, r, w),
        host, port
    )
    
    addr = server.sockets[0].getsockname()
    print(f'[LISTENING] Server is listening on {addr}')

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    PORT = 8080  
    SERVER = get_network_ip()

    print("[STARTING] server is starting...")
    asyncio.run(start_server(SERVER, PORT))