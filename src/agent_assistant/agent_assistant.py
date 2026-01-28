import streamlit as st
from langchain.prompts import PromptTemplate
# from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def init_page() -> None:
    st.set_page_config(
        page_title="Personal Coding Assistant"
    )
    st.header("Personal Coding Assistant")
    st.sidebar.title("Options")

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant for coding. Reply your answer in markdown format.")
        ]
        st.session_state.costs = []

def get_answer(chain, messages) -> tuple[str, float]:
    return chain.invoke(messages)

def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")

def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]

def main() -> None:
    #model = ChatOllama(model = "llama3.1:8b", temperature=0, base_url='http://host.docker.internal:11434')
    model = ChatOllama(model = "llama3.1:8b", temperature=0)

    template = f'''
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        Just answer the question for Python. Format the response as Markdown with code block marked with triple backticks.
        <|eot_id|>
        <|start_header_id|>
        user
        <|end_header_id|>
        Here is the context.
        Context: {{input}}
        <|eot_id|>
        <|start_header_id|>
        assistant
        <|end_header_id|>
    '''
    prompt = PromptTemplate(
        input_variables=["input"],
        template=template
    )
    chain = (
        {"input": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    init_page()
    init_messages()

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("typing ..."):
            answer = get_answer(chain, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    prompt_template = PromptTemplate.from_template(
        "You are a helpful AI assistant. Reply your answer in mardkown format. {prompt}"
    )
    
if __name__=="__main__":
    main()
