# Mycomind Daemon: Advanced Mixture-of-Memory-RAG-Agents (MoMRA) Cognitive Assistant

Mycomind Daemon is a cutting-edge implementation of the Mixture-of-Memory-RAG-Agents (MoMRA) concept, inspired by the vast, interconnected networks of fungal mycelium. This innovative system combines multiple AI models to create a cognitive network that processes information and manages tasks in the background, much like a daemon process in computing.

## What is Mycomind Daemon?

Mycomind Daemon draws inspiration from mycelial networks in nature, which process and distribute information across vast underground systems. Similarly, this AI assistant combines the power of multiple Large Language Models (LLMs) with sophisticated memory management, creating a cognitive network that maintains context and information over extended interactions.

## Key Features

- **Mycelial Network of Models**: Integrates multiple AI models, mimicking the interconnected structure of fungal networks.
- **Hyphal Memory System**: Utilizes Core Memory, Archival Memory, and Conversation History for enhanced context retention, like the branching hyphae of mycelium.
- **Spore-like Model Selection**: Users can choose and configure both reference and aggregate models, spreading capabilities across the network.
- **Adaptive Growth Parameters**: Fine-tune generation with customizable temperature, max tokens, and processing rounds.
- **Nutrient-like Information Flow**: Experience fluid, stream-based response generation.
- **Symbiotic User Interface**: Intuitive Gradio interface with an earth-toned theme for a natural interaction experience.
- **Mycelial Communication Modes**: Support for both single-turn and multi-turn conversations, mimicking mycelium's ability to communicate over short and long distances.
- **Environmental Sensing**: Integrated web search capability for up-to-date information retrieval, like mycelium's ability to sense and respond to its environment.

## How It Works

1. User input is processed by multiple reference models, like how mycelium processes environmental stimuli.
2. Each reference model generates its unique response, similar to different parts of a mycelial network reacting to stimuli.
3. An aggregate model combines and refines these responses, acting like the central hub of a mycelial network.
4. The hyphal memory system updates and retrieves relevant information to maintain context.
5. If needed, the environmental sensing function provides additional, current information.
6. This process can be repeated for multiple rounds, enhancing the quality and context-awareness of the final response, much like how mycelial networks continuously adapt and respond to their environment.

## Hyphal Memory System

Mycomind Daemon employs a sophisticated three-tier memory system, inspired by the branching structure of mycelial hyphae:

1. **Core Memory**: The central hyphae, storing essential context about the user, the AI's persona, and a scratchpad for planning.
2. **Archival Memory**: The extensive network of hyphae, archiving general information and events about user interactions for long-term recall.
3. **Conversation History**: The active growing tips of hyphae, maintaining a searchable log of recent interactions for immediate context.

## Setup and Configuration

1. Clone the repository and navigate to the project directory.

2. Install requirements:

   ```shell
   conda create -n moa python=3.10
   conda activate moa
   pip install -r requirements.txt
   ```

## Configuration

Edit the `.env` file to configure the following parameters:

```bash
API_BASE=http://localhost:11434/v1
API_KEY=ollama

API_BASE_2=http://localhost:11434/v1
API_KEY_2=ollama

MAX_TOKENS=4096
TEMPERATURE=0.7
ROUNDS=1

MODEL_AGGREGATE=llama3:70b-instruct-q6_K

MODEL_REFERENCE_1=phi3:latest 
MODEL_REFERENCE_2=llama3:latest
MODEL_REFERENCE_3=phi3:3.8b-mini-instruct-4k-fp16

OLLAMA_NUM_PARALLEL=4  
OLLAMA_MAX_LOADED_MODELS=4
```

## Running the Application

1. Start the Ollama server:

   ```shell
   OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=4 ollama serve
   ```

2. Launch the Gradio interface:

   ```shell
   conda activate moa
   gradio app.py
   ```
   OR Launch the CLI APP:

   ```shell
   conda activate moa
   python omoa.py
   ```


3. Open your web browser and navigate to the URL provided by Gradio (usually http://localhost:7860).

## Advanced Usage

- **Mycelial Model Customization**: Easily switch between different aggregate and reference models to adapt your cognitive network.
- **Hyphal Memory Management**: Utilize core memory functions to append, remove, or replace information for long-term context.
- **Environmental Sensing Integration**: Leverage the integrated web search capability for up-to-date information during conversations.
- **Multi-Turn Mycelial Communication**: Enable context retention for more dynamic and coherent interactions over time.

## Contributing

We welcome contributions to enhance Mycomind Daemon. Feel free to submit pull requests or open issues for discussions on potential improvements.

## License

This project is licensed under the terms specified in the original MoA repository. Please refer to the original source for detailed licensing information.

---

<div align="center">
  <img src="assets/mycomind_daemon.jpg" alt="Mycomind Daemon Concept Visualization" style="width: 100%; max-width: 600px;" />
</div>
