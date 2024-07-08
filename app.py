import gradio as gr
import os
from dotenv import load_dotenv
from omoa import OllamaAgent, OllamaMixtureOfAgents, DEFAULT_PROMPTS, create_default_agents
from MemoryAssistant.memory import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory
from MemoryAssistant.prompts import wrap_user_message_in_xml_tags_json_mode
from llama_cpp_agent.chat_history.messages import Roles

# Load environment variables
load_dotenv()

# Ollama-specific environment variables
os.environ['OLLAMA_NUM_PARALLEL'] = os.getenv('OLLAMA_NUM_PARALLEL', '4')
os.environ['OLLAMA_MAX_LOADED_MODELS'] = os.getenv('OLLAMA_MAX_LOADED_MODELS', '4')

MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

# Modify these lines to include all available models
ALL_MODELS = [MODEL_AGGREGATE, MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]
ALL_MODELS = [model for model in ALL_MODELS if model]  # Remove any None values

# Global variables to store the MoA configuration
moa_config = {
    "aggregate_agent": None,
    "reference_agents": [],
    "mixture": None
}

# Initialize memory components
agent_core_memory = AgentCoreMemory(["persona", "user", "scratchpad"], core_memory_file="MemoryAssistant/core_memory.json")
agent_retrieval_memory = AgentRetrievalMemory()
agent_event_memory = AgentEventMemory()

def create_mixture():
    moa_config["mixture"] = OllamaMixtureOfAgents(
        moa_config["reference_agents"],
        moa_config["aggregate_agent"]
    )

    # Set the memory components after initialization
    moa_config["mixture"].agent_core_memory = agent_core_memory
    moa_config["mixture"].agent_retrieval_memory = agent_retrieval_memory
    moa_config["mixture"].agent_event_memory = agent_event_memory

def initialize_moa():
    global moa_config
    default_agents = create_default_agents()
    moa_config["aggregate_agent"] = default_agents["SynthesisAgent"]
    moa_config["reference_agents"] = [
        default_agents["AnalyticalAgent"],
        default_agents["HistoricalContextAgent"],
        default_agents["ScienceTruthAgent"]
    ]
    create_mixture()
    print("Mixture of Agents initialized successfully!")

# Call initialize_moa() at the start of the application
initialize_moa()

def create_agent(model, name, system_prompt, **params):
    return OllamaAgent(model, name, system_prompt, **params)

def clear_core_memory():
    if isinstance(moa_config["mixture"], OllamaMixtureOfAgents):
        return moa_config["mixture"].clear_core_memory()
    else:
        return "Error: MoA not initialized properly."

def clear_archival_memory():
    if isinstance(moa_config["mixture"], OllamaMixtureOfAgents):
        return moa_config["mixture"].clear_archival_memory()
    else:
        return "Error: MoA not initialized properly."

def edit_archival_memory(old_content, new_content):
    if isinstance(moa_config["mixture"], OllamaMixtureOfAgents):
        return moa_config["mixture"].edit_archival_memory(old_content, new_content)
    else:
        return "Error: MoA not initialized properly."

async def process_message(message, history):
    # Add user message to event memory
    agent_event_memory.add_event(Roles.user, wrap_user_message_in_xml_tags_json_mode(message))
    
    response, web_search_performed = await moa_config["mixture"].get_response(message)
    
    # Ensure the response is a list of tuples
    if isinstance(response, str):
        formatted_response = [(None, response)]
    elif isinstance(response, list):
        formatted_response = [(None, str(item)) for item in response]
    else:
        formatted_response = [(None, str(response))]
    
    info = f"Generated response using {len(moa_config['reference_agents'])} reference agents and 1 aggregate agent."
    if web_search_performed:
        info += " Web search was performed during response generation."
    
    return formatted_response, info

async def chat(message, history):
    response, processing_info = await process_message(message, history)
    
    # Ensure the response is a list of lists
    formatted_response = [[message, item[1]] if isinstance(item, tuple) else [message, str(item)] for item in response]
    
    # Append the new messages to the history
    updated_history = history + formatted_response
    
    # Ensure the final output is a list of lists
    final_output = [[msg, resp] for msg, resp in updated_history]
    
    return final_output, processing_info


def get_model_params(model_name):
    # Define custom parameters for each model
    params = {
        "llama2": ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"],
        "mistral": ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"],
        "codellama": ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"],
    }
    return params.get(model_name, ["temperature", "top_p", "top_k", "repeat_penalty", "num_ctx"])  # Default parameters if model not found

def update_model_params(model_name):
    params = get_model_params(model_name)
    components = [gr.Markdown(f"### {model_name} Parameters")]
    for param in params:
        if param == "temperature":
            components.append(gr.Slider(minimum=0, maximum=2, value=0.7, step=0.1, label="Temperature"))
        elif param == "top_p":
            components.append(gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="Top P"))
        elif param == "top_k":
            components.append(gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top K"))
        elif param == "repeat_penalty":
            components.append(gr.Slider(minimum=0.1, maximum=2, value=1.1, step=0.05, label="Repeat Penalty"))
        elif param == "num_ctx":
            components.append(gr.Slider(minimum=128, maximum=4096, value=2048, step=128, label="Context Length"))
    
    return gr.Group.update(visible=True, children=components)

def update_agent_config(old_agent_name, model, new_name, prompt, **params):
    new_agent = create_agent(model, new_name, prompt, **params)
    
    if old_agent_name == "SynthesisAgent":
        moa_config["aggregate_agent"] = new_agent
    else:
        moa_config["reference_agents"] = [agent for agent in moa_config["reference_agents"] if agent.name != old_agent_name]
        moa_config["reference_agents"].append(new_agent)
    
    create_mixture()
    return f"Updated agent configuration: {old_agent_name} -> {new_name}"

def edit_core_memory(section, key, value):
    agent_core_memory.update_core_memory(section, {key: value})
    return f"Core memory updated: {section}.{key} = {value}"

def search_archival_memory(query):
    results = agent_retrieval_memory.search(query)
    return f"Archival memory search results for '{query}':\n{results}"

def add_to_archival_memory(content):
    agent_retrieval_memory.insert(content)  # Changed from add to insert
    return f"Added to archival memory: {content}"

def create_gradio_interface():
    theme = gr.themes.Base(
        primary_hue="green",
        secondary_hue="orange",  # Changed from "brown" to "orange"
        neutral_hue="gray",
        font=("Helvetica", "sans-serif"),
    ).set(
        body_background_fill="linear-gradient(to right, #1a2f0f, #3d2b1f)",
        body_background_fill_dark="linear-gradient(to right, #0f1a09, #261a13)",
        button_primary_background_fill="#3d2b1f",
        button_primary_background_fill_hover="#4e3827",
        block_title_text_color="#d3c6aa",
        block_label_text_color="#b8a888",
        input_background_fill="#f0e6d2",
        input_background_fill_dark="#2a1f14",
        input_border_color="#7d6d58",
        input_border_color_dark="#5c4c3d",
        checkbox_background_color="#3d2b1f",
        checkbox_background_color_selected="#5e4534",
        slider_color="#7d6d58",
        slider_color_dark="#5c4c3d",
    )

    css = """
    .gradio-container {
        background-image: url('file/assets/mycelium_bg.png');
        background-size: cover;
        background-attachment: fixed;
    }
    .gr-box {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    .gr-button {
        border-radius: 25px;
    }
    .gr-input {
        border-radius: 10px;
    }
    .gr-form {
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            """
            # Mycomind Daemon: Advanced Mixture-of-Agents (MoA) Cognitive Assistant
            
            Harness the power of interconnected AI models inspired by mycelial networks.
            """
        )
        
        with gr.Tab("Configure MoA"):
            agent_tabs = ["Agent1", "Agent2", "Agent3", "Synthesis Agent"]
            all_agents = moa_config["reference_agents"] + [moa_config["aggregate_agent"]]
            for i, agent in enumerate(all_agents):
                with gr.Tab(agent_tabs[i]):
                    with gr.Row():
                        with gr.Column(scale=1):
                            model = gr.Dropdown(
                                choices=ALL_MODELS,
                                value=agent.model,
                                label="Model"
                            )
                            name = gr.Textbox(
                                value=agent.name,
                                label="Agent Name",
                                interactive=True
                            )
                        
                        with gr.Column(scale=2):
                            prompt = gr.Textbox(
                                value=agent.system_prompt,
                                label="System Prompt",
                                lines=10,
                                interactive=True
                            )
                    
                    with gr.Group() as params_group:
                        gr.Markdown(f"### {agent.model} Parameters")
                        temperature = gr.Slider(minimum=0, maximum=2, value=0.7, step=0.1, label="Temperature")
                        top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="Top P")
                        top_k = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top K")
                        repeat_penalty = gr.Slider(minimum=0.1, maximum=2, value=1.1, step=0.05, label="Repeat Penalty")
                        num_ctx = gr.Slider(minimum=128, maximum=4096, value=2048, step=128, label="Context Length")
                    
                    model.change(
                        update_model_params,
                        inputs=[model],
                        outputs=[params_group]
                    )
                    
                    update_btn = gr.Button(f"Update {agent_tabs[i]}")
                    update_status = gr.Textbox(label="Update Status", interactive=False)
                    
                    def update_agent_wrapper(agent_index):
                        params = {
                            "temperature": temperature.value,
                            "top_p": top_p.value,
                            "top_k": top_k.value,
                            "repeat_penalty": repeat_penalty.value,
                            "num_ctx": num_ctx.value
                        }
                        return update_agent_config(all_agents[agent_index].name, model.value, name.value, prompt.value, **params)
                    
                    update_btn.click(
                        lambda: update_agent_wrapper(i),
                        outputs=[update_status]
                    )
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat History", height=400)
            with gr.Row():
                msg = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2, scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat")
            processing_log = gr.Textbox(label="Processing Log", interactive=False)
        
        with gr.Tab("Memory Management"):
            with gr.Row():
                with gr.Column():
                    core_section = gr.Textbox(label="Core Memory Section")
                    core_key = gr.Textbox(label="Core Memory Key")
                    core_value = gr.Textbox(label="Core Memory Value")
                    edit_core_btn = gr.Button("Edit Core Memory")
                    core_status = gr.Textbox(label="Core Memory Status", interactive=False)
                    
                    # Add a new button to clear core memory
                    clear_core_btn = gr.Button("Clear Core Memory")
                    clear_core_status = gr.Textbox(label="Clear Core Memory Status", interactive=False)
                
                with gr.Column():
                    archival_query = gr.Textbox(label="Archival Memory Search Query")
                    search_archival_btn = gr.Button("Search Archival Memory")
                    archival_results = gr.Textbox(label="Archival Memory Results", interactive=False)
                
                with gr.Column():
                    archival_content = gr.Textbox(label="Content to Add to Archival Memory")
                    add_archival_btn = gr.Button("Add to Archival Memory")
                    archival_status = gr.Textbox(label="Archival Memory Status", interactive=False)
                
                with gr.Column():
                    upload_file = gr.File(label="Upload Document")
                    upload_btn = gr.Button("Process Document")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### Archival Memory Management")
                    clear_archival_btn = gr.Button("Clear Archival Memory")
                    clear_archival_status = gr.Textbox(label="Clear Archival Memory Status", interactive=False)
                    
                    gr.Markdown("### Edit Archival Memory")
                    old_content = gr.Textbox(label="Old Content")
                    new_content = gr.Textbox(label="New Content")
                    edit_archival_btn = gr.Button("Edit Archival Memory")
                    edit_archival_status = gr.Textbox(label="Edit Archival Memory Status", interactive=False)
        
        msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, processing_log])
        send_btn.click(chat, inputs=[msg, chatbot], outputs=[chatbot, processing_log])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, processing_log])
        
        edit_core_btn.click(
            edit_core_memory,
            inputs=[core_section, core_key, core_value],
            outputs=[core_status]
        )
        
        search_archival_btn.click(
            search_archival_memory,
            inputs=[archival_query],
            outputs=[archival_results]
        )
        
        add_archival_btn.click(
            add_to_archival_memory,
            inputs=[archival_content],
            outputs=[archival_status]
        )

        clear_core_btn.click(
            clear_core_memory,
            outputs=[clear_core_status]
        )

        upload_btn.click(
            lambda file: moa_config["mixture"].upload_document(file.name) if file else "No file selected",
            inputs=[upload_file],
            outputs=[upload_status]
        )

        clear_archival_btn.click(
            clear_archival_memory,
            outputs=[clear_archival_status]
        )

        edit_archival_btn.click(
            edit_archival_memory,
            inputs=[old_content, new_content],
            outputs=[edit_archival_status]
        )

    return demo

if __name__ == "__main__":
    initialize_moa()
    demo = create_gradio_interface()
    demo.queue()
    demo.launch()
