import csv

links = {}
with open("links.csv", mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) >= 2:
            links[row[0]] = row[1]

print("\033c", end="")

from bs4 import BeautifulSoup
import requests
import google.generativeai as genai
import streamlit as st

generation_config = {
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 512,
  "response_mime_type": "text/plain"
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

def get_context_from_link(topic: str):
    link = links[topic]
    res = requests.get(link)
    soup = BeautifulSoup(res.text, "html.parser")
    paragraphs = soup.find_all("p")
    info = ""
    for p in paragraphs:
        info += p.text
    return info, link

genai.configure(api_key=st.secrets["general"]["google_api_key"])
model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config, safety_settings=safety_settings, tools=[get_context_from_link])

starter_prompt = f"""
You are an assistant designed to answer questions about the JIS high school handbook.

1. Your task is to first attempt to match the user's prompt to one of the following topics: {list(links.keys())}.
2. If the prompt matches one of these topics, you must call the `get_context_from_link` function with the topic as the argument.
    - The function will return two strings: (1) `info`, the relevant content from the handbook, and (2) `link`, the citation link.
    - Use this information to generate your response.
3. Start your response with 'FROM THE JIS HS HANDBOOK (the citation link): '. Provide a brief and accurate summary of the content from `info`. Make sure your response only contains information directly answering the user prompt, do not provide context on the user prompt.
4. If the prompt does not match any of the topics, or if the prompt matches a topic, but the info from the `get_context_from_link` does not contain the answer to the user prompt, respond briefly without using the info from the function.

IMPORTANT: Do not respond directly on your own if there is a topic match. Always wait for the output from the `get_context_from_link` function before replying.
"""

st.set_page_config(page_title="JIS HS Handbook AI", page_icon="🤖")
st.title("JIS high school handbook AI")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat = model.start_chat(enable_automatic_function_calling=True)
    st.session_state.chat.send_message(starter_prompt)

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question: "):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        res = st.session_state.chat.send_message(prompt)
        st.write(res.text)
    st.session_state.history.append({"role": "assistant", "content": res.text})
