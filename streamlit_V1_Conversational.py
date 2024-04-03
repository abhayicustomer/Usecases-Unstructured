import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from operator import itemgetter

from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

import pinecone
from pinecone import Pinecone
import time
from streamlit_chat import message
import ast

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
    
    # rag_chain_from_docs = (
        # RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        # .assign(context=itemgetter("docs") | RunnableLambda(format_docs))
        # .assign(answer=qa_prompt | llm | StrOutputParser())
        # .pick(["answer", "docs"])
    # )     
      
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    embeddings = OpenAIEmbeddings()
    vectordb = PineconeVectorStore.from_existing_index("talkable-index", embeddings)
    general_system_template_withhistory = r""" 
    Assistant helps the company employees with their company/feedback/reviews/social media related questions, and questions about the company. Be brief in your answers.
    Answer only if the answer is present in the SOURCES below. If answer is not present in the sources don't answer that question.
    Answer ONLY from the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below.
    If there are multiple answers, answer in separate sections.
    Customers can be from Glassdoor, g2, linkedin, facebook, twitter, instagram, google, website, press etc.
    online forums, community and social media means data from Facebook, Twitter, Linkedin, glassdoor, g2, google, website, press etc.
    Don't include the duplicate results.
    
    Given a chat history and the latest user question which might reference context in the chat history.

    Chat History:
    {chat_history}
    
    Question: {input}

    <context>
    {context}
    </context>
   
    """

    qa_prompt_withhistory = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(general_system_template_withhistory)])

    retriever = vectordb.as_retriever(search_type='similarity_score_threshold',search_kwargs={"k":6, "score_threshold": 0.75})

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_withhistory)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    start_time = time.time()
    # Output using the ConversationChain object and the user input, 
    # and storing them in the session
    try:
        with open('chat_history.txt', "r") as f:
            chat_history = f.readlines()
            f.close()
        # chat_history = ast.literal_eval(chat_history)
    except:
        chat_history = []
    if user_input:
        if len(chat_history)>1:
            output = rag_chain.invoke({"input": user_input, "chat_history": chat_history[-4:]})
        else:
            output = rag_chain.invoke({"input": user_input, "chat_history": []})
        elapsed_time = time.time() - start_time
        
        chat_history_old = chat_history[-4:]
        chat_history.extend([HumanMessage(content=user_input), output["answer"]])
        with open('chat_history.txt', "w") as f:
            # f.write(str(chat_history))
            f.write(user_input)
            f.write('\n')
            f.write(output["answer"].replace('\n','...'))
            f.write('\n')
            f.close()

        sources = [i.metadata['source'] for i in output['context']][:5]
        sources = list(set(sources))
        if 'm sorry' in output['answer'].lower():
            sources = []
    
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output['answer']+'\n\nDATA SOURCES: Unstrucured\n'+'\n\nSOURCES:\n'+str('\n'.join(sources))+'\n\nHISTORY:\n'+str(chat_history_old))
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