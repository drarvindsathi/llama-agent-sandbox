import streamlit as st
import pandas as pd
import time
import json
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load environment variables from the .env file
load_dotenv()

api_key = os.environ.get('API_KEY')  # Ensure you have set this environment variable

# Initialize LLM

llm_client = ChatCompletionsClient(
    endpoint="https://datacatalyst-foundry.services.ai.azure.com/models",
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

# llama_model_name = "Llama-4-Scout-17B-16E-Instruct"

# Initialize Azure Blob Storage client
connection_string = os.getenv("AZURE_DB_CONNECTION_STRING")
container_name = "sandbox"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def get_blob_content(blob_name):
    """
    Retrieve the content of a blob from Azure Blob Storage.
    
    :param blob_name: Name of the blob to retrieve
    :return: Content of the blob as a string
    """
    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().content_as_text()
        return json.loads(blob_data)
    except Exception as e:
        st.error(f"Error retrieving blob content: {e}")
        return None



# Authentication function
def authenticate_user(email):
    try:
        return email.strip().endswith('@kpmg.com')
    except Exception as e:
        st.error(f"Email string invalid: {e}")
        return False



def edit_agent_component(agent):
    st.subheader(f"✏️ Display Agent: {agent['name']}")

    # Editable context
    edited_agent_str = json.dumps(agent, indent=2)
        # Parse JSON and handle syntax errors
    # Text area for JSON input/editing
    edited_agent_str = st.text_area("Edit the JSON below:", value=edited_agent_str, height=400)

    # Parse JSON and handle syntax errors
    try:
        edited_agent = json.loads(edited_agent_str)
        st.success("Successfully parsed JSON!")

        # Display parsed JSON data
        st.json(edited_agent)

        # Convert dictionary back to JSON string with indentation
        json_to_download = json.dumps(edited_agent, indent=4)

        # Download button
        st.download_button(
            label="Download JSON",
            data=json_to_download,
            file_name="agent_config.json",
            mime="application/json"
        )
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {e}")

    # Model switcher
    # updated_model = st.selectbox(
    #     "Switch Model",
    #     options=["Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8"],
    #     index=["Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8"].index(agent["model"]),
    #     key=f"model_{agent['name']}",
    # )

    # Optional file upload
    updated_files = st.file_uploader(
        "Upload New Files (optional)",
        accept_multiple_files=True,
        key=f"files_{agent['name']}",
    )

    # Update button
    if st.button("Update Agent", key=f"update_{agent['name']}"):
        agent = edited_agent
        # if updated_model:
        #     agent["model"] = updated_model
        if updated_files:
            agent["files"] = updated_files
        st.session_state["show_popup"] = False
        st.session_state["mode"] = "use_existing"
        st.rerun()



def get_embedding(text):
    try:
        response = gpt_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []


def get_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0



def retrieve_top_descriptions(prompt, model_choice, description_vectors, top_n=3):
    query_embedding = get_embedding(prompt)
    similarity_list = []

    for record in description_vectors:
        similarity = get_similarity(query_embedding, record["embedding"])
        similarity_list.append(
            {
                "index": record["index"],
                "description": record["description"],
                "similarity": similarity,
            }
        )

    sorted_descriptions = sorted(
        similarity_list, key=lambda x: x["similarity"], reverse=True
    )
    return sorted_descriptions[:top_n]



def retrieve_similar_chunks(prompt, top_n=3):
    query_embedding = get_embedding("###Question: " + prompt + "###")
    similarity_json_list = []
    for record in st.session_state.get("embedding_vector", []):
        similarity = get_similarity(query_embedding, record["embedding"])
        similarity_json_list.append(
            {"index": record["index"], "similarity": similarity}
        )
    sorted_embedding = sorted(
        similarity_json_list, key=lambda x: x["similarity"], reverse=True
    )
    for item in sorted_embedding[:top_n]:
        chunk_text = st.session_state["double_pages"][item["index"]]
        item["text"] = (
            chunk_text if isinstance(chunk_text, str) else json.dumps(chunk_text)
        )
    return sorted_embedding[:top_n]


def make_embedded_prompt(prompt, top_n=3):
    top_chunks = retrieve_similar_chunks(prompt, top_n)
    reported_content = json.dumps(top_chunks)
    return f"Query - {prompt}\nSearch Results - {reported_content}"

# exclude keys from JSON - remooves specified keys from each dictionary in a list of dictionaries
def exclude_keys(json_list, keys_to_exclude):
    """
    Creates a new JSON list excluding a specified key and its value from each dictionary.

    Args:
      json_list: The original JSON list (list of dictionaries).
      key_to_exclude: The key to exclude from each dictionary.

    Returns:
      A new JSON list with the specified key excluded, or the original list if no changes were made.
    """
    new_json_list = []
    for item in json_list:
        new_item = {k: v for k, v in item.items() if k not in keys_to_exclude}
        new_json_list.append(new_item)
    return new_json_list

# extract_text - retrieves extracted text from original JSON data for chosen articles   
def extract_text(chosen_articles, original_json_data, id_column, text_column):
    """Retrieves extracted text from original JSON data for chosen articles."""
    extracted_texts = []
    
    # Create a dictionary for fast lookups of URLs and their corresponding extracted text.
    original_data_dict = {item[id_column]: item.get(text_column, None) for item in original_json_data}

    for article in chosen_articles:
        id = article[id_column]
        extracted_text = original_data_dict.get(id)  # Efficiently retrieve the text
        if extracted_text:
            extracted_texts.append({id_column: id, text_column: extracted_text})
        else:
            extracted_texts.append({id_column: id, text_column: None}) # Or handle missing text differently.
            st.write(f"Warning: No extracted text found for {id_column}: {id}")

    return extracted_texts

def clean_llm_output(output):
    """
    Takes a string output from the LLM and strips off any characters
    before the first '[' and after the last ']'.

    :param output: The LLM output string
    :return: Cleaned string
    """
    # Find the position of the first '[' and the last ']'
    start_index = output.find('[')
    end_index = output.rfind(']') + 1

    # If both '[' and ']' are found, return the substring
    if start_index != -1 and end_index != -1:
        cleaned_output = output[start_index:end_index]
    else:
        # Return original output if brackets are not found
        cleaned_output = output

    return cleaned_output

# Agents are defined below.  They use two different system prompts and a knowledge source to answer user queries. The output of 

class Agent:
    def __init__(self, system_prompt_agent_1, system_prompt_agent_2,  knowledge_source_json, agent_1_columns, model):
        """
        Initialize an Agent object with the given attributes.

        :param system_prompt_agent_1: String prompt for agent 1
        :param system_prompt_agent_2: String prompt for agent 2
        :param knowledge_source: Path to the knowledge source file (a JSON file)
        :param agent_1_columns: A list of keys corresponding to columns relevant to agent 1
        :param client: API client for chat completions
        :param model_selected: Model to be used for chat completions
        """
        self.system_prompt_agent_1 = system_prompt_agent_1
        self.system_prompt_agent_2 = system_prompt_agent_2
        self.knowledge_source_json = knowledge_source_json  # Placeholder for knowledge source JSON
        self.agent_1_columns = agent_1_columns
        self.model = model

    def agent_1_execution(self, user_query):
        """
        Execute logic for agent 1.

        :param user_query: Query from the user
        :return: Selected documents as a response string
        """


        # Filter the knowledge source based on the columns
        summary_json = exclude_keys(self.knowledge_source_json, self.agent_1_columns)
        summary_json_text = json.dumps(summary_json, indent=2)

        # Prepare chat memory
        chat_memory = [
            SystemMessage(content=self.system_prompt_agent_1 + f"Here is the information provided:\n{summary_json_text}"),
            UserMessage(content=user_query)
        ]

        # Request response from the chat API
        response = llm_client.complete(
            messages=chat_memory,
            model=self.model,
            temperature = 0
        )

        # Extract selected documents
        selected_documents = response.choices[0].message.content
        cleaned_documents = clean_llm_output(selected_documents)
        try:
            selected_json =  json.loads(cleaned_documents)
        except:
            # Handle the case where the output is not valid JSON
            # This could be due to the LLM output not matching the expected format
            # Log the error or handle it as needed
            st.write(f"Error: Unable to parse selected documents. Expected JSON format but could not locate JSON: {selected_documents}")
            selected_json = []
        return selected_json

    def agent_2_execution(self, user_query, selected_documents, id_column, text_column):
        """
        Execute logic for agent 2.

        :param user_query: Query from the user
        :param selected_documents: Documents selected from agent 1 execution
        :return: Query response as a string
        """
        # Extract text from selected documents
        selected_documents_text = extract_text(selected_documents, self.knowledge_source_json, id_column, text_column)
        if not selected_documents_text:
            st.error("No text extracted from selected documents.")
            return 

        # Prepare chat memory
        chat_memory = [
            SystemMessage(content=self.system_prompt_agent_2 + f"Here is the information provided: \n{selected_documents_text}"),
            UserMessage(content=user_query)
        ]

        # Request response from the chat API
        response = llm_client.complete(
            messages=chat_memory,
            model=self.model,
            temperature = 0.5
        )

        # Extract query response
        query_response = response.choices[0].message.content
        return query_response

# Main dashboard
def main_dashboard():
    for key in [
        "agents",
        "chat_history",
        "selected_agent",
        "mode",
        "show_popup",
        "transitioning",
        "show_tlka_chat",
        "tlka_chat_history",
        "models",
        "model",
        "selecting_model"
    ]:
        # Set agents and chat history to empty if present
        if key not in st.session_state:
            st.session_state[key] = [] if key in ["chat_history"] else {} if key in ["agents"] else None

    # Add premade Thought Leadership agent if not already added
    if "thought_leadership_agent" not in st.session_state:
        tlka_agent_json = {"id": "thought_leadership_agent",
                           "name": "Thought Leadership Knowledge Assistant",
                        "description": "An AI assistant that helps users understand KPMG's point of view on  specific topics.",
                        "system_prompt_agent_1": """You are a Thought Leadership Knowledge Assistant and you assist the team in understanding and getting to know KPMG Point of View on specific topics.

        Take the user query and use the information provided below to find top 5 relevant article.
        Provide your response by listing JSONs matched.  Your response should only include JSON and nothing else.""",
                            "system_prompt_agent_2": """You are a Thought Leadership Knowledge Assistant and you assist the team in understanding and getting to know KPMG Point of View on specific topics.
        Use the information provided to answer the user query.  
        Your response should be in the form of a point of view on the topic, using the selected content as context and cite URLs of the articles used.""",
                            "knowledge_source": "TLKA_Knowledge.json" ,
                            "agent_1_columns": ['Extracted-Text', 'Entities', 'Keywords', 'JSON'],
                            "id_column": "URL",
                            "text_column": "Extracted-Text"}

        st.session_state["agents"]["thought_leadership_agent"] = tlka_agent_json

    if "signals_agent" not in st.session_state:
        signals_agent_json = {"id": "signals_agent",
                              "name": "Signals Knowledge Assistant",
                   "description": "An AI assistant that helps users explore KPMG's Signals Repository contents in solving business problems.",
                   "system_prompt_agent_1": """You are a Data Analysis assistant. Users will give you business problems and you will use the information provided below to find top 5 relevant tables.

                    Provide your response by listing JSONs matched.  Your response should only include JSON and nothing else.""",
                            "system_prompt_agent_2": """You are a Data Analysis assistant. Users will give you business problems.
                            Use the information provided to define an analytics approach to solving the problem. Your response should be in the form of an analytics approach to solving the problem, using the selected content as context.""",
                            "knowledge_source": "signals_knowledge.json" ,
                            "agent_1_columns": ['Extracted-text'],
                            "id_column": "elastic_id",
                            "text_column": "Extracted-text"}

        st.session_state["agents"]["signals_agent"] = signals_agent_json

    if st.session_state["transitioning"]:
        with st.spinner("Loading chat interface..."):
            time.sleep(1.5)
        st.session_state["transitioning"] = False
        st.rerun()

    with st.sidebar:

        st.markdown(
            """
        <style>
            .logo-container {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            }
            .logo-container img {
                height: 60px;
            }
        </style>
        <div class="logo-container">
        <img src="https://th.bing.com/th/id/R.3cc79b367ef20d08fb2a0f0c0c48c9c8?rik=sS09IzBwlPT8lQ&riu=http%3a%2f%2flogos-download.com%2fwp-content%2fuploads%2f2016%2f10%2fKPMG_logo.png&ehk=scn63cOoidCDU1lz7FAHvCw%2f3yI9ubmrUTJA7ow6LLQ%3d&risl=&pid=ImgRaw&r=0" alt="Description of the image">
            
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style='display: flex; justify-content: center;'>
                <style>
                    div.stButton > button {
                        width: 100%;
                        max-width: 200px;
                        margin: auto;
                    }
                </style>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Home"):
            st.session_state["selected_agent"] = None
            st.session_state["mode"] = None
            st.session_state["chat_history"] = []
            st.session_state["show_popup"] = False
            st.session_state["transitioning"] = False

        st.markdown("---")

        st.header("Agents")
        if st.session_state["agents"]:
            for agent in list(st.session_state["agents"].values()):
                if st.button(f"🛠️ Display {agent['name']}"):
                    st.session_state["selected_agent"] = agent
                    st.session_state["mode"] = "edit_agent"
                    st.session_state["show_popup"] = True
                    st.rerun()
        else:
            st.info("No agents available. Please create a new model.")

    st.title("Agent Management Dashboard")

    st.session_state["models"]=["Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8"]
    


    if st.session_state["mode"] is None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use an Existing Agent"):
                st.session_state["mode"] = "use_existing"
                st.rerun()

        with col2:
            if st.button("Create a New Model"):
                st.session_state["show_popup"] = True
                st.session_state["mode"] = "create_new"
                st.rerun()
    else:
        if st.button("Change Mode"):
            st.session_state["mode"] = None
            st.rerun()

    if st.session_state["show_popup"]:
        if st.session_state["mode"] == "edit_agent":

            edit_agent_component(st.session_state["selected_agent"])
        else:
            st.subheader("🛠️ Create a New Agent")
            agent_name = st.text_input("Agent Name")
            model_choice = st.selectbox("Model", ["LLaMA Scout", "LLaMA Maverick", "GPT-4"])
            context = st.text_area("Context / Instructions")
            uploaded_files = st.file_uploader(
                "Upload Files", accept_multiple_files=True
            )
            if st.button("Create Agent"):
                new_agent = {
                    "name": agent_name,
                    "model": model_choice,
                    "context": context,
                    "files": uploaded_files,
                }
                st.session_state["agents"].append(new_agent)
                st.session_state["selected_agent"] = new_agent
                st.session_state["chat_history"] = []
                st.session_state["show_popup"] = False
                st.session_state["mode"] = "create_new"
                st.session_state["transitioning"] = True

                # Vectorize uploaded files

                if uploaded_files:
                    embedding_result = vectorize_uploaded_files(
                        uploaded_files, model_choice
                    )
                    if embedding_result:
                        embedding_vector, double_pages = embedding_result
                        st.session_state["embedding_vector"] = embedding_vector
                        st.session_state["double_pages"] = double_pages

                st.rerun()

    if (st.session_state["mode"] == "use_existing" and not st.session_state["selected_agent"]):
        st.subheader("Select an Existing Agent")
        for agent in list(st.session_state["agents"].values()):
            if st.button(f"{agent['name']}", key="select_"+agent['name']):
                st.session_state["selected_agent"] = agent
                st.session_state["chat_history"] = []
                st.rerun()

    if st.session_state["selected_agent"] and st.session_state["mode"] in [
        "use_existing",
        "create_new",
    ]:
            st.subheader(f"💬 Chat with {st.session_state['selected_agent']['name']}")
            model_selection_label = "Select Model" if not st.session_state["model"] else "Change Model"
            if not st.session_state["selecting_model"] and st.button(model_selection_label):
                st.session_state["selecting_model"] = True
            if st.session_state.get("selecting_model", False):
                for model in st.session_state["models"]:
                    if st.button(f"Use {model} model", key="use_model_"+model):
                        st.session_state["selecting_model"] = False
                        st.session_state["model"] = model
                        st.session_state["chat_history"] = []
                        st.rerun()
            if st.session_state["model"]:
                st.subheader("Selected model: " + st.session_state["model"])
                for message in st.session_state["chat_history"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                user_input = st.chat_input("Ask your agent something...")
                if user_input:
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": user_input}
                    )
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    context = st.session_state["selected_agent"]["system_prompt_agent_1"]

                    model_choice = st.session_state["model"]

                    id_column = st.session_state["selected_agent"]["id_column"]
                    text_column = st.session_state["selected_agent"]["text_column"]

                    agent_json = st.session_state["selected_agent"]

                    try:
                        if model_choice in st.session_state["models"]:
                            knowledge_source_json = get_blob_content(agent_json["knowledge_source"])
                            st.write("Knowledge Source JSON:", knowledge_source_json[0])
                            current_agent = Agent(system_prompt_agent_1=agent_json["system_prompt_agent_1"],
                                system_prompt_agent_2=agent_json["system_prompt_agent_2"],
                                knowledge_source_json=knowledge_source_json,
                                agent_1_columns=agent_json["agent_1_columns"],
                                model=model_choice)
                            selected_docs = current_agent.agent_1_execution(user_input)
                            st.write("Selected Documents:", selected_docs)
                            if selected_docs:
                                response = current_agent.agent_2_execution(user_input, selected_docs, id_column, text_column)
                                assistant_reply = response
                            else:
                                assistant_reply = "⚠️ No relevant documents found to answer your query."

                        else:
                            assistant_reply = f"⚠️ Unknown model selected: {model_choice}"

                    except Exception as e:
                        assistant_reply = f"⚠️ Error generating response: {e}"

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": assistant_reply}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(assistant_reply)
            



# Entry point
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "email_input" not in st.session_state:
        st.session_state["email_input"] = ""

    if not st.session_state["authenticated"]:
        st.title("Authentication Required")
        st.session_state["email_input"] = st.text_input(
            "Enter your email", value=st.session_state["email_input"]
        )
        if st.button("Submit"):
            if authenticate_user(st.session_state["email_input"]):

                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid email. Please try again.")
    else:
        main_dashboard()


main()
