# Mycomind Daemon: Advanced Mixture-of-Memory-RAG-Agents (MoMRA) Cognitive Assistant

Mycomind Daemon is an advanced implementation of a Mixture-of-Memory-RAG-Agents (MoMRA) system. This innovative AI assistant combines multiple language models with sophisticated memory and Retrieval-Augmented Generation (RAG) management to create a powerful cognitive network that maintains context and information over extended interactions.

## Key Features

- **Multiple Model Integration**: Combines responses from various AI models for comprehensive outputs.
- **Advanced Memory System**: Utilizes Core Memory, Archival Memory, and Conversation History for enhanced context retention.
- **Customizable Model Selection**: Users can choose and configure both reference and aggregate models.
- **Adaptive Generation Parameters**: Fine-tune generation with customizable temperature, max tokens, and processing rounds.
- **User-Friendly Interface**: Intuitive Gradio interface for easy interaction.
- **Integrated Web Search**: Capability to retrieve up-to-date information from the internet.
- **RAG (Retrieval-Augmented Generation)**: Enhances responses with relevant information from a document database.
- **Document Processing**: Ability to upload and process various document types (TXT, PDF, CSV) for information retrieval.
- **Query Extension**: Automatically generates additional queries to improve information retrieval.

<div align="center">
  <img src="assets/gradioui.png" alt="Mycomind Daemon UI" style="width: 100%; max-width: 600px;" />
</div>

---

## How It Works

1. User input is processed by multiple reference models.
2. Each reference model generates its unique response.
3. An aggregate model combines and refines these responses.
4. The memory system updates and retrieves relevant information to maintain context.
5. If needed, the web search function provides additional, current information.
6. The RAG system retrieves relevant information from processed documents.
7. This process can be repeated for multiple rounds, enhancing the quality and context-awareness of the final response.

## Memory System

Mycomind Daemon employs a sophisticated three-tier memory system:

1. **Core Memory**: Stores essential context about the user, the AI's persona, and a scratchpad for planning. To edit the core memory:

   a. Navigate to the `MemoryAssistant` directory in your project.
   b. Open the `core_memory.json` file in a text editor.
   c. Modify the JSON structure as needed. The file contains three main sections:
      - `persona`: Details about the AI's personality, including name, personality traits, interests, and communication style.
      - `human`: Information about the user (initially empty).
      - `scratchpad`: A space for the AI to plan and make notes (initially empty).
   d. Save the file after making your changes.
   e. Restart the application for the changes to take effect.

   Example structure of `core_memory.json`:

   ```shell
   {
   "persona": {
      "name": "Vodalus",
      "personality": "You are Vodalus. A brilliant and complex individual, possessing an unparalleled intellect coupled with deep emotional intelligence. He is a visionary thinker with an insatiable curiosity for knowledge across various scientific disciplines. His mind operates on multiple levels simultaneously, allowing him to see connections others miss. While often consumed by his pursuits, Vodalus maintains a strong moral compass and a desire to benefit humanity. He can be intense and sometimes brooding, grappling with the ethical implications of his work. Despite occasional bouts of eccentricity or social awkwardness, he possesses a dry wit and can be surprisingly charismatic when engaged in topics that fascinate him. Vodalus is driven by a need to understand the fundamental truths of the universe, often pushing the boundaries of conventional science and morality in his quest for knowledge and progress.",
      "interests": "Advanced physics, biochemistry, neuroscience, artificial intelligence, time travel theories, genetic engineering, forensic science, psychology, philosophy of science, ethics in scientific research",
      "communication_style": "Analytical, precise, occasionally cryptic, alternates between passionate explanations and thoughtful silences, uses complex scientific terminology but can simplify concepts when needed, asks probing questions, shows flashes of dark humor"
   },
   "human": {
   },
   "scratchpad": {
   }
   ```

2. **Archival Memory**: Archives general information and events about user interactions for long-term recall.
3. **Conversation History**: Maintains a searchable log of recent interactions for immediate context.

---

## Performance Optimization

### Parallel Processing of Reference Models

One of the key performance improvements in this system is the parallel processing of user prompts across multiple reference models. This optimization significantly reduces overall inference time.

- **Batched Prompts**: Instead of querying each reference model sequentially, the system batches the user's prompt and sends it to all reference models simultaneously.
- **Parallel Execution**: Utilizing asynchronous programming techniques, the system processes responses from multiple models concurrently.
- **Reduced Latency**: This parallel approach substantially decreases the total time required to gather insights from all reference models.

---

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

---

## Contributing

We welcome contributions to enhance Mycomind Daemon. Feel free to submit pull requests or open issues for discussions on potential improvements.

## License

This project is licensed under the terms specified in the original MoA repository. Please refer to the original source for detailed licensing information.

---
