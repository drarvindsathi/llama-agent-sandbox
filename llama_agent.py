import streamlit as st
import pandas as pd
import time
import json
import os
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential



# Initialize GPT client

subscription_key = "D1NpYdYhrp7LJCxay5UnUy44MKhuHjNdQLmmRN9GNdKI8FIFPnHgJQQJ99BGACYeBjFXJ3w3AAAAACOG0n4n"

gpt_client = AzureOpenAI(
    azure_endpoint="https://datacatalyst-foundry.services.ai.azure.com/models",
    api_key=subscription_key,
    api_version="2024-12-01-preview"
)

# Initialize LLaMA client
llama_client = ChatCompletionsClient(
    endpoint="https://datacatalyst-foundry.services.ai.azure.com/models",
    credential=AzureKeyCredential(subscription_key),
    api_version="2024-05-01-preview"
)
llama_model_name = "Llama-4-Scout-17B-16E-Instruct"

# Authentication function
def authenticate_user(email):
    try:
        df = pd.read_csv("kpmg_emails.csv")
        df['email'] = df['email'].str.lower()
        return email.lower() in df['email'].values
    except Exception as e:
        st.error(f"Error reading kpmg_emails.csv: {e}")
        return False

# Main dashboard
def main_dashboard():
    for key in ["agents", "chat_history", "selected_agent", "mode", "show_popup", "transitioning"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key in ["agents", "chat_history"] else None

    # Add premade Thought Leadership agent if not already added
    if "thought_leadership_added" not in st.session_state:
        thought_leadership_agent = {
            "name": "Thought Leadership Knowledge Assistant",
            "model": "LLaMA Scout",
            "context": """You are a Thought Leadership Knowledge Assistant and you assist the team in understanding and getting to know KPMG Point of View on specific topics.

Skills: You will use the Topic Classification Table to classify the topic of your query and pdf_files_result.xlsx to find answers first and then embellish the responses with weblinks provided in the summary.json or using the other articles.

Detailed Steps:
1. Take the user query and classify using Topic Classification Table. Provide your classification back to the user and get confirmation.
2. Use pdf_files_result to find related articles. List articles found and get confirmation from the user.
3. Find appropriate article in the collection and provide a detailed response using the content of the article.
""",
            "files": None
        }
        st.session_state["agents"].append(thought_leadership_agent)
        st.session_state["thought_leadership_added"] = True

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
        unsafe_allow_html=True
    )
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
        
     


        st.header("LLM Capabilities")
        st.markdown("""
        - **GPT-4**: Versatile and accurate — best for complex reasoning and strategic tasks
        - **LLaMA Maverick**: Enterprise-grade and contextual — ideal for summarization and decision support
        - **LLaMA Scout**: Fast and agile — great for quick lookups and document navigation
        """)

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
            st.subheader(f"✏️ Edit Agent: {st.session_state['selected_agent']['name']}")
            context = st.text_area("Edit Context / Instructions", value=st.session_state["selected_agent"]["context"])
            uploaded_files = st.file_uploader("Upload New Files (optional)", accept_multiple_files=True)
            if st.button("Update Agent"):
                st.session_state["selected_agent"]["context"] = context
                if uploaded_files:
                    st.session_state["selected_agent"]["files"] = uploaded_files
                st.session_state["show_popup"] = False
                st.session_state["mode"] = "use_existing"
                st.rerun()
        else:
            st.subheader("🛠️ Create a New Agent")
            agent_name = st.text_input("Agent Name")
            model_choice = st.selectbox("Model", ["LLaMA Scout", "GPT-4"])
            context = st.text_area("Context / Instructions")
            uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
            if st.button("Create Agent"):
                new_agent = {
                    "name": agent_name,
                    "model": model_choice,
                    "context": context,
                    "files": uploaded_files
                }
                st.session_state["agents"].append(new_agent)
                st.session_state["selected_agent"] = new_agent
                st.session_state["chat_history"] = []
                st.session_state["show_popup"] = False
                st.session_state["mode"] = "create_new"
                st.session_state["transitioning"] = True
                st.rerun()

    if st.session_state["mode"] == "use_existing" and not st.session_state["selected_agent"]:
        st.subheader("Select an Existing Agent")
        for agent in st.session_state["agents"]:
            if st.button(f"{agent['name']} ({agent['model']})"):
                st.session_state["selected_agent"] = agent
                st.session_state["chat_history"] = []

    if st.session_state["selected_agent"] and st.session_state["mode"] in ["use_existing", "create_new"]:
        st.subheader(f"💬 Chat with {st.session_state['selected_agent']['name']}")
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask your agent something...")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            context = st.session_state["selected_agent"]["context"]
            file_contents = ""
            if "files" in st.session_state["selected_agent"] and st.session_state["selected_agent"]["files"]:
                for file in st.session_state["selected_agent"]["files"]:
                    try:
                        file_contents += file.read().decode("utf-8") + "\n"
                    except:
                        pass

            system_prompt = context
            if file_contents:
                system_prompt += f"\n\nAdditional reference material:\n{file_contents}"

            model_choice = st.session_state["selected_agent"]["model"]

            try:
                if model_choice == "LLaMA Scout":
                    response = llama_client.complete(
                        messages=[
                            SystemMessage(content=system_prompt),
                            UserMessage(content=user_input)
                        ],
                        max_tokens=2048,
                        temperature=0.8,
                        top_p=0.1,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        model=llama_model_name
                    )
                    assistant_reply = response.choices[0].message.content

                elif model_choice == "GPT-4":
                    conversations = [{"role": "system", "content": system_prompt}] + [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state["chat_history"]
                    ]
                    response = gpt_client.chat.completions.create(
                        model="gpt-4.1",
                        temperature=0,
                        messages=conversations
                    )
                    assistant_reply = response.choices[0].message.content

                else:
                    assistant_reply = f"⚠️ Unknown model selected: {model_choice}"

            except Exception as e:
                assistant_reply = f"⚠️ Error generating response: {e}"

            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})
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
        st.session_state["email_input"] = st.text_input("Enter your email", value=st.session_state["email_input"])
        if st.button("Submit"):
            if authenticate_user(st.session_state["email_input"]):
                
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid email. Please try again.")
    else:
        main_dashboard()


main()
