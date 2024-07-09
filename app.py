import gradio as gr
import os
import json
from dotenv import load_dotenv
from omoa import OllamaAgent, OllamaMixtureOfAgents, DEFAULT_PROMPTS, create_default_agents
from MemoryAssistant.memory import AgentCoreMemory, AgentEventMemory
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
agent_event_memory = AgentEventMemory()

def create_mixture():
    moa_config["mixture"] = OllamaMixtureOfAgents(
        moa_config["reference_agents"],
        moa_config["aggregate_agent"]
    )

    # Set the memory components after initialization
    moa_config["mixture"].agent_core_memory = agent_core_memory
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
    moa_config["mixture"] = OllamaMixtureOfAgents(
        moa_config["reference_agents"],
        moa_config["aggregate_agent"],
        temperature=0.6,
        max_tokens=2048,
        rounds=1
    )
    moa_config["mixture"].web_search_enabled = True  
    moa_config["mixture"].agent_core_memory = agent_core_memory
    moa_config["mixture"].agent_event_memory = agent_event_memory
    print("Mixture of Agents initialized successfully!")

# Call initialize_moa() at the start of the application
initialize_moa()

def create_agent(model, name, system_prompt, **params):
    supported_params = ['model', 'name', 'system_prompt']  # Add any other supported parameters here
    filtered_params = {k: v for k, v in params.items() if k in supported_params}
    return OllamaAgent(model, name, system_prompt, **filtered_params)

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


def update_memory(self, message, role):
    # Update event memory
    self.agent_event_memory.add_event(role, message)

    # Update RAG
    self.rag.add_document(message)

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
    
    return components

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
    results = moa_config["mixture"].search_archival_memory(query)
    return f"Archival memory search results for '{query}':\n{results}"

def add_to_archival_memory(content):
    if isinstance(moa_config["mixture"], OllamaMixtureOfAgents):
        moa_config["mixture"].add_to_archival_memory(content)
        return f"Added to archival memory: {content}"
    return f"Failed to add to archival memory: {content}. MoA not initialized properly."

def toggle_web_search(enabled):
    if isinstance(moa_config["mixture"], OllamaMixtureOfAgents):
        return moa_config["mixture"].toggle_web_search(enabled)
    return "Error: MoA not initialized properly."




def create_gradio_interface():
    global moa_config
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
            # Mycomind Daemon: Advanced Mixture-of-Memory-RAG-Agents (MoMRA) Cognitive Assistant
            
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
                    archival_query = gr.Textbox(label="Archival Memory Search Query")
                    search_archival_btn = gr.Button("Search Archival Memory")
                    archival_results = gr.Textbox(label="Archival Memory Results", interactive=False)

                with gr.Column():
                    gr.Markdown("### Archival Memory Management")
                    clear_archival_btn = gr.Button("Clear Archival Memory")
                    clear_archival_status = gr.Textbox(label="Clear Archival Memory Status", interactive=False)
                    
                    gr.Markdown("### Edit Archival Memory")
                    old_content = gr.Textbox(label="Old Content")
                    new_content = gr.Textbox(label="New Content")
                    edit_archival_btn = gr.Button("Edit Archival Memory")
                    edit_archival_status = gr.Textbox(label="Edit Archival Memory Status", interactive=False)

                with gr.Column():
                    archival_content = gr.Textbox(label="Content to Add to Archival Memory")
                    add_archival_btn = gr.Button("Add to Archival Memory")
                    archival_status = gr.Textbox(label="Archival Memory Status", interactive=False)

                # with gr.Row():
                #     gr.Markdown("### Core Memory Viewer")
                #     core_memory_viewer = gr.JSON(label="Current Core Memory", value=moa_config["mixture"].load_core_memory())
                #     refresh_core_memory_btn = gr.Button("Refresh Core Memory View")

                # with gr.Row():
                #     gr.Markdown("### Core Memory Editor")
                #     core_memory_editor = gr.Textbox(label="Edit Core Memory", value=json.dumps(moa_config["mixture"].load_core_memory(), indent=2), lines=10, max_lines=20)
                #     update_core_memory_btn = gr.Button("Update Core Memory")
                #     core_memory_status = gr.Textbox(label="Core Memory Update Status", interactive=False)
                

                
        with gr.Tab("RAG Management"):
            with gr.Row():
                with gr.Column():        
                    upload_file = gr.File(label="Upload Document")
                    upload_btn = gr.Button("Process Document")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### RAG Configuration")
                    chunk_size = gr.Slider(minimum=128, maximum=1024, value=512, step=64, label="Chunk Size")
                    chunk_overlap = gr.Slider(minimum=0, maximum=256, value=0, step=32, label="Chunk Overlap")
                    k_value = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Retrieved Documents (k)")
            
            with gr.Row():
                gr.Markdown("### RAG Status")
                rag_status = gr.JSON(label="Current RAG Status")
                refresh_rag_status_btn = gr.Button("Refresh RAG Status")

            def update_rag_config(chunk_size, chunk_overlap, k_value):
                rag = moa_config["mixture"].rag
                
                # Update attributes if they exist
                if hasattr(rag, 'chunk_size'):
                    rag.chunk_size = chunk_size
                if hasattr(rag, 'chunk_overlap'):
                    rag.chunk_overlap = chunk_overlap
                if hasattr(rag, 'k'):
                    rag.k = k_value
                
                # If there's a specific method to update configuration, use it
                if hasattr(rag, 'update_config'):
                    rag.update_config(chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k_value)
                
                # If there's a method to reinitialize the index with new settings, call it
                if hasattr(rag, 'reinitialize_index'):
                    rag.reinitialize_index()
                
                return "RAG configuration updated successfully"

            def get_rag_status():
                rag = moa_config["mixture"].rag
                status = {
                    "Index Size": rag.get_index_size() if hasattr(rag, 'get_index_size') else "Not available",
                    "Current Configuration": rag.get_config() if hasattr(rag, 'get_config') else "Not available"
                }
                
                # Try to get document count if the method exists
                if hasattr(rag, 'get_document_count'):
                    status["Document Count"] = rag.get_document_count()
                elif hasattr(rag, 'index') and hasattr(rag.index, '__len__'):
                    status["Document Count"] = len(rag.index)
                else:
                    status["Document Count"] = "Not available"
                
                return status

            update_rag_config_btn = gr.Button("Update RAG Configuration")
            update_rag_config_status = gr.Textbox(label="Update Status", interactive=False)

            update_rag_config_btn.click(
                update_rag_config,
                inputs=[chunk_size, chunk_overlap, k_value],
                outputs=[update_rag_config_status]
            )

            refresh_rag_status_btn.click(
                get_rag_status,
                outputs=[rag_status]
            )

        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Web Search")
                    web_search_toggle = gr.Checkbox(label="Enable Web Search", value=True)
                    web_search_status = gr.Textbox(label="Web Search Status", interactive=False)

                with gr.Column():
                    gr.Markdown("### Processing Parameters")
                    rounds_slider = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Processing Rounds")
                    temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                    max_tokens_slider = gr.Slider(minimum=100, maximum=4096, value=1000, step=100, label="Max Tokens")

            with gr.Row():
                gr.Markdown("### Additional Settings")
                stream_output_toggle = gr.Checkbox(label="Stream Output", value=True)
                debug_mode_toggle = gr.Checkbox(label="Debug Mode", value=False)

            #def refresh_core_memory():
            #    return moa_config["mixture"].load_core_memory()

            #def update_core_memory(new_core_memory_str):
            #    try:
            #        new_core_memory = json.loads(new_core_memory_str)
            #        moa_config["mixture"].core_memory = new_core_memory
            #        moa_config["mixture"].agent_core_memory.update_core_memory(new_core_memory)
            #        moa_config["mixture"].agent_core_memory.save_core_memory(moa_config["mixture"].core_memory_file)
            #        return json.dumps(new_core_memory, indent=2), "Core memory updated successfully"
            #    except json.JSONDecodeError:
            #        return json.dumps(moa_config["mixture"].load_core_memory(), indent=2), "Error: Invalid JSON format"
            #    except Exception as e:
            #        return json.dumps(moa_config["mixture"].load_core_memory(), indent=2), f"Error updating core memory: {str(e)}"

            def update_settings(rounds, temperature, max_tokens, stream_output, debug_mode):
                moa_config["mixture"].rounds = rounds
                moa_config["mixture"].temperature = temperature
                moa_config["mixture"].max_tokens = max_tokens
                moa_config["mixture"].stream_output = stream_output
                moa_config["mixture"].debug_mode = debug_mode
                return "Settings updated successfully"

            # update_core_memory_btn.click(
            #     update_core_memory,
            #     inputs=[core_memory_editor],
            #     outputs=[core_memory_status]
            # )

            # refresh_core_memory_btn.click(
            #     refresh_core_memory,
            #     outputs=[core_memory_viewer]
            # )

            # update_core_memory_btn.click(
            #     update_core_memory,
            #     inputs=[core_memory_editor],
            #     outputs=[core_memory_viewer, core_memory_status]
            # )

            settings_update_btn = gr.Button("Update Settings")
            settings_update_status = gr.Textbox(label="Settings Update Status", interactive=False)

            settings_update_btn.click(
                update_settings,
                inputs=[rounds_slider, temperature_slider, max_tokens_slider, stream_output_toggle, debug_mode_toggle],
                outputs=[settings_update_status]
            )

            web_search_toggle.change(
                toggle_web_search,
                inputs=[web_search_toggle],
                outputs=[web_search_status]
            )

        msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, processing_log])
        send_btn.click(chat, inputs=[msg, chatbot], outputs=[chatbot, processing_log])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, processing_log])
        
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
    demo.launch(share=True)
