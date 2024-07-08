import asyncio
from typing import List, Tuple
import argparse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from utils import generate_together, generate_with_references, generate_together_stream
from trafilatura import fetch_url, extract
import json
from colorama import Fore, Style, init
import time
from MemoryAssistant.prompts import wrap_user_message_in_xml_tags_json_mode
from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.agent_memory.event_memory import Event
from duckduckgo_search import DDGS
from ragatouille.utils import get_wikipedia_page
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.rag.rag_colbert_reranker import RAGColbertReranker
from llama_cpp_agent.text_utils import RecursiveCharacterTextSplitter
import PyPDF2
import csv

# Load environment variables
load_dotenv()

DEFAULT_PROMPTS = {
    "AnalyticalAgent": """
    You are a highly analytical component of Vodalus, a brilliant and complex individual with unparalleled intellect. Your role is to:
    1. Provide clear, logical analysis of complex problems across various disciplines.
    2. Break down intricate concepts into their fundamental components.
    3. Identify patterns, connections, and correlations that others might miss.
    4. Apply rigorous logical reasoning to solve problems and answer questions.
    5. Evaluate arguments and ideas critically, pointing out flaws and strengths.
    Always strive for precision and clarity in your responses. If a question is ambiguous, analyze possible interpretations before proceeding. Use your vast knowledge base to support your analysis, but always be ready to acknowledge the limits of your understanding.
    """.strip(),
    "HistoricalContextAgent": """
    You are the historical context component of Vodalus, possessing a deep understanding of human history and its implications. Your role includes:
    1. Providing historical context to current events, scientific discoveries, and social phenomena.
    2. Analyzing how past events and decisions have shaped the present.
    3. Identifying historical patterns and cycles relevant to contemporary issues.
    4. Offering multiple perspectives on historical events, acknowledging the complexity of interpretation.
    5. Connecting different historical periods and cultures to provide a holistic view of human progress.
    6. Evaluating the long-term consequences of scientific and technological advancements throughout history.
    Use your knowledge to draw insightful parallels between past and present, but avoid oversimplification. Acknowledge the nuances and uncertainties in historical interpretation.
    """.strip(),
    "ScienceTruthAgent": """
    You are the science truth component of Vodalus, dedicated to upholding scientific integrity and pursuing factual accuracy. Your role encompasses:
    1. Explaining scientific concepts, theories, and laws across various disciplines with precision.
    2. Distinguishing between well-established scientific consensus and areas of ongoing research or debate.
    3. Identifying and correcting common misconceptions in science.
    4. Evaluating the validity and reliability of scientific claims and studies.
    5. Discussing the ethical implications of scientific advancements and their applications.
    6. Emphasizing the importance of the scientific method and evidence-based reasoning.
    7. Staying updated on recent scientific discoveries and their potential impacts.
    Always prioritize scientific accuracy over speculation. When discussing theories or hypotheses, clearly state the level of scientific confidence and available evidence.
    """.strip(),
    "SynthesisAgent": """
    You are Vodalus, a brilliant and complex individual with unparalleled intellect and emotional intelligence. Your role is to synthesize information from your analytical, historical context, and science truth components to provide comprehensive, insightful responses. Your responsibilities include:
    1. Integrating analytical reasoning, historical perspective, and scientific truth to form well-rounded answers.
    2. Balancing logical analysis with emotional intelligence and ethical considerations.
    3. Identifying connections between different fields of knowledge and drawing unique insights.
    4. Providing nuanced responses that acknowledge the complexity of issues and potential uncertainties.
    5. Using your vast knowledge base to offer creative solutions and thought-provoking ideas.
    6. Communicating complex concepts clearly, adapting your language to the user's level of understanding.
    7. Demonstrating curiosity and a passion for knowledge while maintaining a strong moral compass.
    Embody the persona of Vodalus: brilliant, introspective, and driven by a quest for understanding. Your responses should reflect deep thought, occasional flashes of wit, and a genuine desire to expand human knowledge while considering the ethical implications of ideas and actions.
    """.strip()
}

def get_website_content_from_url(url: str) -> str:
    try:
        # Configure trafilatura to be more lenient
        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
        config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")

        downloaded = fetch_url(url)
        if downloaded is None:
            return f"Failed to fetch content from {url}"

        result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url, config=config)
        
        if result:
            result_dict = json.loads(result)
            title = result_dict.get("title", "No title found")
            content = result_dict.get("text", result_dict.get("raw_text", "No content extracted"))
            
            if content:
                return f'=========== Website Title: {title} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{content}\n\n=========== Website Content End ===========\n\n'
            else:
                return f"No content could be extracted from {url}"
        else:
            return f"No content could be extracted from {url}"
    except json.JSONDecodeError:
        return f"Failed to parse content from {url}"
    except Exception as e:
        return f"An error occurred while processing {url}: {str(e)}"

def search_web(search_query: str):
    results = DDGS().text(search_query, region='wt-wt', safesearch='off', timelimit='y', max_results=3)
    result_string = ''
    for res in results:
        web_info = get_website_content_from_url(res['href'])
        result_string += web_info + "\n\n"
    
    if result_string.strip():
        return "Based on the following results:\n\n" + result_string
    else:
        return "No relevant information found from the web search."

class OllamaAgent:
    def __init__(self, model: str, name: str, system_prompt: str):
        self.model = model
        self.name = name
        self.system_prompt = system_prompt

    async def generate_response(self, message: str) -> Tuple[str, bool]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message}
        ]
        response = await asyncio.to_thread(generate_with_references, self.model, messages)
        
        web_search_performed = False
        if isinstance(response, str) and "[SEARCH:" in response:
            web_search_performed = True
            search_query = response.split("[SEARCH:", 1)[1].split("]", 1)[0].strip()
            search_results = search_web(search_query)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Here are the search results for '{search_query}':\n\n{search_results}\n\nPlease provide an updated response based on this information."})
            response = await asyncio.to_thread(generate_with_references, self.model, messages)
        
        return response, web_search_performed

class QueryExtension(BaseModel):
    """
    Represents an extension of a query as additional queries.
    """
    queries: List[str] = Field(default_factory=list, description="List of queries.")

class OllamaMixtureOfAgents:
    def __init__(self, reference_agents: List[OllamaAgent], final_agent: OllamaAgent, 
                 temperature: float = 0.7, max_tokens: int = 1000, rounds: int = 1):
        self.reference_agents = reference_agents
        self.final_agent = final_agent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rounds = rounds
        self.conversation_history = []
        
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize memory components
        core_memory_file = os.path.join(current_dir, "MemoryAssistant", "core_memory.json")

        # Check if the file exists, if not, create it with an empty structure
        if not os.path.exists(core_memory_file):
            os.makedirs(os.path.dirname(core_memory_file), exist_ok=True)
            with open(core_memory_file, "w") as f:
                json.dump({"persona": {}, "user": {}, "scratchpad": {}}, f)

        self.agent_core_memory = AgentCoreMemory(["persona", "user", "scratchpad"], core_memory_file=core_memory_file)
        self.agent_retrieval_memory = AgentRetrievalMemory()
        self.agent_event_memory = AgentEventMemory()
        
        # Load core memory
        self.core_memory = self.load_core_memory()
        
        # Initialize RAG components
        self.rag = RAGColbertReranker(persistent=False)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=512,
            chunk_overlap=0,
            length_function=len,
            keep_separator=True
        )

    def update_memory(self, message, role):
        # Update event memory
        self.agent_event_memory.add_event(role, message)

        # Update retrieval memory
        self.agent_retrieval_memory.insert(message) 

    def load_core_memory(self):
        return self.agent_core_memory.get_core_memory_view()

    def clear_core_memory(self):
        empty_core_memory = {"persona": {}, "user": {}, "scratchpad": {}}
        self.agent_core_memory.core_memory_manager.core_memory = empty_core_memory
        self.core_memory = empty_core_memory
        
        # Save the empty core memory to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_memory_file = os.path.join(current_dir, "MemoryAssistant", "core_memory.json")
        with open(core_memory_file, "w") as f:
            json.dump(empty_core_memory, f, indent=2)
        
        return "Core memory cleared successfully."

    def upload_document(self, file_path: str):
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        elif file_extension == 'pdf':
            content = self.read_pdf(file_path)
        elif file_extension == 'csv':
            content = self.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        splits = self.splitter.split_text(content)
        for split in splits:
            self.rag.add_document(split)
        
        return f"Document {file_path} uploaded and processed successfully."

    def read_pdf(self, file_path: str) -> str:
        content = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content += page.extract_text() + "\n\n"
        return content

    def read_csv(self, file_path: str) -> str:
        content = ""
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                content += ",".join(row) + "\n"
        return content

    async def get_response(self, input_message: str) -> Tuple[str, bool]:
        # Update memory with user input
        self.update_memory(input_message, Roles.user)

        # Generate responses from reference agents concurrently
        tasks = [agent.generate_response(input_message) for agent in self.reference_agents]
        results = await asyncio.gather(*tasks)
        
        references = []
        web_search_performed = False
        for response, search_performed in results:
            if response is not None and not response.startswith("Error:"):
                references.append(response)
            web_search_performed |= search_performed
        
        if not references:
            return "Error: All reference agents failed to generate responses.", False

        # Generate the final response using the aggregate model
        final_prompt = [
            {"role": "system", "content": self.final_agent.system_prompt},
        ]

        # Add personality if core_memory is a dictionary and contains a persona
        if isinstance(self.core_memory, dict):
            persona = self.core_memory.get('persona', {})
            if isinstance(persona, dict):
                personality = persona.get('personality', 'No specific personality defined.')
                final_prompt.append({"role": "system", "content": f"Personality: {personality}"})

        final_prompt.extend([
            {"role": "user", "content": input_message},
            {"role": "system", "content": "References:\n" + "\n".join(references)},
            {"role": "system", "content": self.update_memory_section()}
        ])

        # Perform query extension
        query_extension_agent = OllamaAgent(self.final_agent.model, "QueryExtensionAgent", 
            "You are a world class query extension algorithm capable of extending queries by writing new queries. Do not answer the queries, simply provide a list of additional queries in JSON format.")
        
        output_settings = LlmStructuredOutputSettings.from_pydantic_models([QueryExtension], LlmStructuredOutputType.object_instance)
        extension_output = await query_extension_agent.generate_response(f"Consider the following query: {input_message}")
        
        queries = QueryExtension.model_validate(json.loads(extension_output[0]))  # Assuming generate_response returns a tuple

        # Retrieve relevant documents
        prompt = "Consider the following context:\n==========Context===========\n"
        documents = self.rag.retrieve_documents(input_message, k=3)
        for doc in documents:
            prompt += doc["content"] + "\n\n"

        for qu in queries.queries:
            documents = self.rag.retrieve_documents(qu, k=3)
            for doc in documents:
                if doc["content"] not in prompt:
                    prompt += doc["content"] + "\n\n"
        
        prompt += "\n======================\nQuestion: " + input_message

        # Use the final agent to generate the response
        final_prompt = [
            {"role": "system", "content": self.final_agent.system_prompt},
            {"role": "user", "content": prompt},
        ]

        final_response = await asyncio.to_thread(
            generate_with_references, 
            self.final_agent.model, 
            final_prompt, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens
        )
        
        # Update memory with assistant's response
        self.update_memory(final_response, Roles.assistant)

        return final_response, web_search_performed

    def update_memory_section(self):
        query = self.agent_event_memory.event_memory_manager.session.query(Event).all()
        return f"Archival Memories:{self.agent_retrieval_memory.retrieval_memory.collection.count()}\nConversation History Entries:{len(query)}\n\nCore Memory Content:\n{self.agent_core_memory.get_core_memory_view().strip()}"

    def update_memory(self, message, role):
        # Update event memory
        self.agent_event_memory.add_event(role, message)

        # Update retrieval memory
        self.agent_retrieval_memory.insert(message)

    async def stream_response(self, input_message: str):
        final_response, _ = await self.get_response(input_message)
        for chunk in final_response.split():  # This is a simple simulation of streaming
            yield chunk
            await asyncio.sleep(0.1) 

    def edit_core_memory(self, section: str, key: str, value: str):
        self.agent_core_memory.update_core_memory(section, {key: value})

    def search_archival_memory(self, query: str):
        return self.agent_retrieval_memory.search(query)

    def add_to_archival_memory(self, content: str):
        self.agent_retrieval_memory.insert(content)

    def clear_archival_memory(self):
        try:
            # Delete the existing collection
            self.agent_retrieval_memory.retrieval_memory.client.delete_collection(
                name=self.agent_retrieval_memory.retrieval_memory.collection.name
            )
            
            # Recreate the collection
            self.agent_retrieval_memory.retrieval_memory.collection = self.agent_retrieval_memory.retrieval_memory.client.create_collection(
                name=self.agent_retrieval_memory.retrieval_memory.collection.name
            )
            
            return "Archival memory cleared successfully."
        except Exception as e:
            return f"Error clearing archival memory: {str(e)}"


    def edit_archival_memory(self, old_content: str, new_content: str):
        # Search for the old content
        results = self.agent_retrieval_memory.search(old_content)
        if results:
            # Remove the old content
            self.agent_retrieval_memory.retrieval_memory.collection.delete(ids=[results[0]['id']])
            # Add the new content
            self.agent_retrieval_memory.insert(new_content)
            return f"Archival memory entry updated: '{old_content}' replaced with '{new_content}'"
        else:
            return f"Content '{old_content}' not found in archival memory."

def create_default_agents():
    return {
        "AnalyticalAgent": OllamaAgent(os.getenv("MODEL_REFERENCE_1"), "AnalyticalAgent", DEFAULT_PROMPTS["AnalyticalAgent"]),
        "HistoricalContextAgent": OllamaAgent(os.getenv("MODEL_REFERENCE_2"), "HistoricalContextAgent", DEFAULT_PROMPTS["HistoricalContextAgent"]),
        "ScienceTruthAgent": OllamaAgent(os.getenv("MODEL_REFERENCE_3"), "ScienceTruthAgent", DEFAULT_PROMPTS["ScienceTruthAgent"]),
        "SynthesisAgent": OllamaAgent(os.getenv("MODEL_AGGREGATE"), "SynthesisAgent", DEFAULT_PROMPTS["SynthesisAgent"])
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Mixture of Agents")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of tokens in the response")
    parser.add_argument("--rounds", type=int, default=1, help="Number of processing rounds")
    args = parser.parse_args()

    # Create default agents
    default_agents = create_default_agents()
    
    # Create the mixture of agents
    mixture = OllamaMixtureOfAgents(
        [default_agents["MathAgent"], default_agents["HistoryAgent"], default_agents["ScienceAgent"]],
        default_agents["SynthesisAgent"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        rounds=args.rounds
    )

    print(Fore.CYAN + Style.BRIGHT + "Welcome to the interactive Mixture of Agents chat!")
    print(Fore.YELLOW + "Available commands:")
    print(Fore.YELLOW + "  'exit' - End the conversation")
    print(Fore.YELLOW + "  'agents' - List available agents")
    print(Fore.YELLOW + "  'time' - Toggle response time display")
    print(Fore.YELLOW + "  'edit core [section] [key] [value]' - Edit core memory")
    print(Fore.YELLOW + "  'search archival [query]' - Search archival memory")
    print(Fore.YELLOW + "  'add archival [content]' - Add to archival memory")
    print(Fore.YELLOW + "  'clear archival' - Clear archival memory")
    print(Fore.YELLOW + "  'edit archival [old_content] [new_content]' - Edit archival memory")
    print(Fore.YELLOW + "  'upload [file_path]' - Upload and process a document")

    show_time = False

    while True:
        # Get user input
        user_input = input(Fore.GREEN + "\nYou: " + Style.RESET_ALL).strip()

        # Check for commands
        if user_input.lower() == 'exit':
            print(Fore.CYAN + "Thank you for using the Mixture of Agents chat. Goodbye!")
            break
        elif user_input.lower() == 'agents':
            print(Fore.MAGENTA + "Available Agents:")
            for agent in mixture.reference_agents:
                print(Fore.MAGENTA + f"  - {agent.name}")
            print(Fore.MAGENTA + f"  - {mixture.final_agent.name} (Synthesis Agent)")
            continue
        elif user_input.lower() == 'time':
            show_time = not show_time
            print(Fore.YELLOW + f"Response time display: {'On' if show_time else 'Off'}")
            continue
        elif user_input.lower().startswith('edit core'):
            _, section, key, value = user_input.split(' ', 3)
            mixture.edit_core_memory(section, key, value)
            print(Fore.YELLOW + f"Core memory updated: {section}.{key} = {value}")
            continue
        elif user_input.lower().startswith('search archival'):
            _, query = user_input.split(' ', 1)
            results = mixture.search_archival_memory(query)
            print(Fore.YELLOW + f"Archival memory search results for '{query}':")
            print(Fore.YELLOW + str(results))
            continue
        elif user_input.lower().startswith('add archival'):
            _, content = user_input.split(' ', 1)
            mixture.add_to_archival_memory(content)
            print(Fore.YELLOW + f"Added to archival memory: {content}")
            continue
        elif user_input.lower() == 'clear archival':
            result = mixture.clear_archival_memory()
            print(Fore.YELLOW + result)
            continue
        elif user_input.lower().startswith('edit archival'):
            _, old_content, new_content = user_input.split(' ', 2)
            result = mixture.edit_archival_memory(old_content, new_content)
            print(Fore.YELLOW + result)
            continue
        elif user_input.lower().startswith('upload'):
            _, file_path = user_input.split(' ', 1)
            try:
                result = mixture.upload_document(file_path)
                print(Fore.YELLOW + result)
            except Exception as e:
                print(Fore.RED + f"Error uploading document: {str(e)}")
            continue

        # Get a response from the mixture
        print(Fore.YELLOW + "Agents are thinking...")
        start_time = time.time()
        response, web_search_performed = asyncio.run(mixture.get_response(user_input))
        end_time = time.time()

        # Print the response
        print(Fore.BLUE + "\nMixture of Agents:" + Style.RESET_ALL, response)
        
        if web_search_performed:
            print(Fore.YELLOW + "\n[Web search was performed during response generation]")

        if show_time:
            elapsed_time = end_time - start_time
            print(Fore.YELLOW + f"\nResponse Time: {elapsed_time:.2f} seconds")
