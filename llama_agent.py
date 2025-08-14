import streamlit as st
import pandas as pd
import time
import json
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

api_key = os.environ.get('API_KEY')  # Ensure you have set this environment variable

# Initialize LLM

llm_client = ChatCompletionsClient(
    endpoint="https://datacatalyst-foundry.services.ai.azure.com/models",
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

llama_model_name = "Llama-4-Scout-17B-16E-Instruct"



# Authentication function
def authenticate_user(email):
    try:
        return email.strip().endswith('@kpmg.com')
    except Exception as e:
        st.error(f"Email string invalid: {e}")
        return False


def edit_agent_component(agent):
    st.subheader(f"✏️ Edit Agent: {agent['name']}")

    # Editable context
    updated_context = st.text_area(
        "Edit Context / Instructions",
        value=agent["context"],
        key=f"context_{agent['name']}",
    )

    # Model switcher
    updated_model = st.selectbox(
        "Switch Model",
        options=["Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8"],
        index=["Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8"].index(agent["model"]),
        key=f"model_{agent['name']}",
    )

    # Optional file upload
    updated_files = st.file_uploader(
        "Upload New Files (optional)",
        accept_multiple_files=True,
        key=f"files_{agent['name']}",
    )

    # Update button
    if st.button("Update Agent", key=f"update_{agent['name']}"):
        agent["context"] = updated_context
        agent["model"] = updated_model
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
def extract_text(chosen_articles, original_json_data):
    """Retrieves extracted text from original JSON data for chosen articles."""
    extracted_texts = []
    
    # Create a dictionary for fast lookups of URLs and their corresponding extracted text.
    original_data_dict = {item['URL']: item.get('Extracted-Text', None) for item in original_json_data}

    for article in chosen_articles:
        url = article['URL']
        extracted_text = original_data_dict.get(url)  # Efficiently retrieve the text
        if extracted_text:
            extracted_texts.append({"URL": url, "Extracted-Text": extracted_text})
        else:
            extracted_texts.append({"URL": url, "Extracted-Text": None}) # Or handle missing text differently.
            st.write(f"Warning: No extracted text found for URL: {url}")

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
            model=self.model
        )

        # Extract selected documents
        selected_documents = response.choices[0].message.content
        selected_documents = clean_llm_output(selected_documents)
        return json.loads(selected_documents)

    def agent_2_execution(self, user_query, selected_documents):
        """
        Execute logic for agent 2.

        :param user_query: Query from the user
        :param selected_documents: Documents selected from agent 1 execution
        :return: Query response as a string
        """
        # Extract text from selected documents
        selected_documents_text = extract_text(selected_documents, self.knowledge_source_json)

        # Prepare chat memory
        chat_memory = [
            SystemMessage(content=self.system_prompt_agent_2 + f"Here is the information provided: \n{selected_documents_text}"),
            UserMessage(content=user_query)
        ]

        # Request response from the chat API
        response = llm_client.complete(
            messages=chat_memory,
            model=self.model
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
    ]:
        # Set agents and chat history to empty if present
        if key not in st.session_state:
            st.session_state[key] = [] if key in ["agents", "chat_history"] else None

    # Add premade Thought Leadership agent if not already added
    if "thought_leadership_added" not in st.session_state:
        tlka_agent_json = {"name": "Thought Leadership Knowledge Assistant",
                        "description": "An AI assistant that helps users understand KPMG's point of view on  specific topics.",
                        "system_prompt_agent_1": """You are a Thought Leadership Knowledge Assistant and you assist the team in understanding and getting to know KPMG Point of View on specific topics.

        Take the user query and use the information provided below to find top 5 relevant article.
        Provide your response by listing JSONs matched.  Your response should only include JSON and nothing else.""",
                            "system_prompt_agent_2": """You are a Thought Leadership Knowledge Assistant and you assist the team in understanding and getting to know KPMG Point of View on specific topics.
        Use the information provided to answer the user query.  
        Your response should be in the form of a point of view on the topic, using the selected content as context and cite URLs of the articles used.""",
                            "knowledge_source": "TLKA_Knowledge.json" ,
                            "agent_1_columns": ['Extracted-Text', 'Entities', 'Keywords', 'JSON'],
        #                    "model": "Llama-4-Maverick-17B-128E-Instruct-FP8"}
                            "model": "Llama-4-Scout-17B-16E-Instruct"}

        with open(tlka_agent_json["knowledge_source"], "r") as file:   
            knowledge_source_json = json.load(file) 

        thought_leadership_agent = Agent(system_prompt_agent_1=tlka_agent_json["system_prompt_agent_1"],
                        system_prompt_agent_2=tlka_agent_json["system_prompt_agent_2"],
                        knowledge_source_json=knowledge_source_json,
                        agent_1_columns=tlka_agent_json["agent_1_columns"],
                        model=tlka_agent_json["model"])
        st.session_state["thought_leadership_agent"] = thought_leadership_agent
        st.session_state["agents"].append(tlka_agent_json)
        st.session_state["thought_leadership_added"] = True
        st.session_state["show_tlka_chat"] = True

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
            for agent in st.session_state["agents"]:
                if st.button(f"🛠️ Edit {agent['name']}"):
                    st.session_state["selected_agent"] = agent
                    st.session_state["mode"] = "edit_agent"
                    st.session_state["show_popup"] = True
                    st.rerun()
        else:
            st.info("No agents available. Please create a new model.")

    st.title("Agent Management Dashboard")

    if st.session_state["mode"] is None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use an Existing Agent"):
                st.session_state["mode"] = "use_existing"

        with col2:
            if st.button("Create a New Model"):
                st.session_state["show_popup"] = True
                st.session_state["mode"] = "create_new"

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

    if (st.session_state["mode"] == "use_existing"
        and not st.session_state["selected_agent"]):
        st.subheader("Select an Existing Agent")
        for agent in st.session_state["agents"]:
            if st.button(f"{agent['name']} ({agent['model']})"):
                st.session_state["selected_agent"] = agent
                st.session_state["chat_history"] = []

    if st.session_state["selected_agent"] and st.session_state["mode"] in [
        "use_existing",
        "create_new",
    ]:
        st.subheader(f"💬 Chat with {st.session_state['selected_agent']['name']}")
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


            model_choice = st.session_state["selected_agent"]["model"]

            try:
                if model_choice == "Llama-4-Scout-17B-16E-Instruct":
                    thought_leadership_agent = st.session_state["thought_leadership_agent"]
                    selected_docs = thought_leadership_agent.agent_1_execution(user_input)
                    response = thought_leadership_agent.agent_2_execution(user_input, selected_docs)
                    assistant_reply = response

                elif model_choice == "LLaMA Maverick":
                    thought_leadership_agent = st.session_state["thought_leadership_agent"]
                    selected_docs = thought_leadership_agent.agent_1_execution(user_input)
                    response = thought_leadership_agent.agent_2_execution(user_input, selected_docs)
                    assistant_reply = response

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
