import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from operator import itemgetter

from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

import pinecone
from pinecone import Pinecone
from streamlit_chat import message

import time

import dotenv
dotenv.load_dotenv()

def main():
    st.title("Talkable Question and Answering System🤖")
    st.markdown(
        ''' 
            > :black[**A ChatGPT based QnA 
            for Unstructured Data(Websites, PDFs)**]
            ''')

    
    # Get the user input
    user_input = st.text_input("You: ", st.session_state["input"],
                                key="input",
                                placeholder="Your Chatbot friend! Ask away ...",
                                label_visibility='hidden')
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    embeddings = OpenAIEmbeddings()
    vectordb = PineconeVectorStore.from_existing_index("talkable-index", embeddings)
    general_system_template = r""" 
    Assistant helps the company employees with their company/feedback/reviews/social media related questions, and questions about the company. Be brief in your answers.
    Answer ONLY from the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below.
    If there are multiple answers, answer in separate sections with document name.
    Customers can be from Glassdoor, g2, linkedin, facebook, twitter, instagram, google, website, press etc.
    online forums, community and social media means data from Facebook, Twitter, Linkedin, glassdoor, g2, google, website, press etc.
    Don't include the duplicate results.

    Question: {question}

    SOURCES:
    {context}
    """

    messages = [HumanMessagePromptTemplate.from_template(general_system_template)]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectordb.as_retriever(search_type='similarity_score_threshold',search_kwargs={"k":10, "score_threshold": 0.7})

    rag_chain_from_docs = (
        RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        .assign(context=itemgetter("docs") | RunnableLambda(format_docs))
        .assign(answer=qa_prompt | llm | StrOutputParser())
        .pick(["answer", "docs"])
    )   

    start_time = time.time()
    
    # Output using the ConversationChain object and the user input, 
    # and storing them in the session
    if user_input:
        output = rag_chain_from_docs.invoke(user_input)
        sources = [i.metadata['source'] for i in output['docs']][:5]
        sources = list(set(sources))
        elapsed_time = time.time() - start_time
        with open('chat_history.txt', "a+") as f:
            f.write(user_input)
            f.write('\n')
            f.write(output["answer"])
            f.write('\n')
            f.close()
    
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output['answer']+'\n\nSOURCES:\n'+str('\n'.join(sources)))
        st.write(f"Response time: {elapsed_time:.2f} seconds")

    # Display the conversation history using an expander
    with st.expander("Conversation", expanded=True):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.info(st.session_state["past"][i], icon="🧐")
            st.success(st.session_state["generated"][i], icon="🤖")


if __name__ == '__main__':
    st.set_page_config(page_title='ChatGPT ChatBot🤖', layout='centered')

    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""

    main()