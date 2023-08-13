import pandas as pd
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


load_dotenv()

def truncate(column):
    return column[:1000]

def preprocess():
    products_df = pd.read_csv('product_data.csv',converters={
                    'bullet_point':truncate,
                    'item_keywords':truncate,
                    'item_name':truncate
                    })
    products_df['item_keywords'].replace('',None,inplace=True)
    products_df.dropna(subset=['item_keywords'],inplace=True)
    products_df.reset_index(drop=True,inplace=True)
    product_metadata = products_df.head(3000).to_dict(orient='index')
    return product_metadata

def vector_store_setup():
    product_metadata = preprocess()
    texts = [v['item_name'] for k,v in product_metadata.items()]
    embeddings = OpenAIEmbeddings()
    vector_store = Redis.from_texts(
        texts = texts,
        metadatas=product_metadata.values(),
        embedding = embeddings,
        index_name="product",
        redis_url="redis://localhost:6379"
    )
    return vector_store

def template_creation():
    template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
    Or end the conversation if it seems like it is done.
    Chat history:\"""
    {chat_history}
    \"""
    Follow up Input: \"""
    {question}
    \"""
    Standalone question:"""

    condense_question_prompt = PromptTemplate.from_template(template)

    template = """You are a friendly, conversational retail shopping assistant. Use the following
    context including product names, descriptions and keywords to show the customer what is available,
    help find what they want and answer any questions.
    Context:\"""
    {context}
    \"""
    Question:
    Helpful Answer:"""

    qa_prompt= PromptTemplate.from_template(template)

    return condense_question_prompt,qa_prompt

def chain_creation():
    condense_question_prompt,qa_prompt = template_creation()

    llm = OpenAI(temperature=0)

    streaming_llm = OpenAI(streaming=True, callback_manager=BaseCallbackManager([
        StreamingStdOutCallbackHandler()
    ]),verbose=True, max_tokens=150, temperature=0)

    question_generator = LLMChain(
        llm=llm,
        prompt=condense_question_prompt
    )

    doc_chain = load_qa_chain(
    llm=streaming_llm,
    chain_type="stuff",
    prompt=qa_prompt
    )

    vector_store = vector_store_setup()

    chatbot = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )

if __name__ == "__main__":
    # create a chat history buffer
    chat_history = []

    chatbot = chain_creation()

    # gather user input for the first question to kick off the bot
    question = input("Hi! What are you looking for today? ")

    while True:
        result = chatbot(
            {"question": question, "chat_history": chat_history}
        )
        print("\nBot: " + result["answer"])
        chat_history.append((question, result["answer"]))
        question = input("You: ")
