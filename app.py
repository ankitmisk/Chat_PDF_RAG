import streamlit as st # type: ignore
import os
from PyPDF2 import PdfReader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain # type: ignore
from langchain.chains import create_retrieval_chain # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
# from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
# load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            with st.spinner(f"Reading {pdf.name}..."):
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"PDF read error: {str(e)}")
        st.error(f"Failed to read PDF: {str(e)}")
        return None

def get_text_chunks(text):
    try:
        if not text:
            raise ValueError("Empty text content")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300
        )
        return text_splitter.split_text(text)
    except Exception as e:
        logging.error(f"Text split error: {str(e)}")
        st.error(f"Document processing failed: {str(e)}")
        return None

def get_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise ValueError("No text chunks to process")
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyDl3dc5U_rBGDuEHiY9JIicnb-6FXd56ms"
        )
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Added error handling for file operations
        try:
            vector_store.save_local("faiss_index")
            return True
        except Exception as e:
            logging.error(f"Index save error: {str(e)}")
            st.error("Failed to save document index")
            return False
            
    except Exception as e:
        logging.error(f"Embedding error: {str(e)}")
        st.error("Failed to create document embeddings")
        return False

def get_conversational_chain():
    try:
        prompt_template = """
        Answer based on the provided context. If the query is unrelated to the context, provide a general response from external knowledge sources.
        
        Context: {context}
        
        Question: {input}
        
        Answer:"""


        # prompt for not revealing system details
        # prompt_template = """
        # Provide answers based on the given context. If the query is unrelated to the provided context, respond with relevant general information without revealing system details.
        
        # Context: {context}
        
        # Question: {input}
        
        # Answer:"""
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-002",
            temperature=0.3,
            google_api_key="AIzaSyDl3dc5U_rBGDuEHiY9JIicnb-6FXd56ms"
        )
        
        prompt = PromptTemplate.from_template(prompt_template)
        return create_stuff_documents_chain(model, prompt)
        
    except Exception as e:
        logging.error(f"Model init error: {str(e)}")
        st.error("AI model initialization failed")
        return None

def user_input(user_question):
    try:
        if not user_question.strip():
            raise ValueError("Empty question")
            
        if not os.path.exists("faiss_index"):
            st.error("Please process PDFs first!")
            return None
            
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key="AIzaSyDl3dc5U_rBGDuEHiY9JIicnb-6FXd56ms"
            )
            vector_store = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.error(f"Index load error: {str(e)}")
            st.error("Failed to load document index")
            return None
            
        retriever = vector_store.as_retriever()
        document_chain = get_conversational_chain()
        
        if not document_chain:
            return None
            
        try:
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": user_question})
            return response.get("answer", "No answer found")
        except Exception as e:
            logging.error(f"Query error: {str(e)}")
            return "Error generating answer"
            
    except Exception as e:
        logging.error(f"Input handling error: {str(e)}")
        return f"Processing error: {str(e)}"


def main():
    try:
        st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
        st.header("Chat with Multiple PDFs 📚")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        with st.sidebar:
            try:
                st.subheader("Document Management")
                pdf_docs = st.file_uploader(
                    "Upload PDF files here", 
                    accept_multiple_files=True,
                    type="pdf"
                )
                
                if st.button("Process PDFs", disabled=not pdf_docs):
                    with st.spinner("Processing documents..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks and get_vector_store(text_chunks):
                                st.success("Processing complete! Ask questions now")
                
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
                    
            except Exception as e:
                logging.error(f"Sidebar error: {str(e)}")
                st.error("Sidebar operation failed")

        # Display chat history at the top
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for chat in st.session_state.chat_history:  # Natural order (oldest first)
                with st.chat_message("user"):
                    st.markdown(chat["question"])
                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])
                st.divider() 

        # Chat input at bottom with latest messages appearing above it
        user_question = st.chat_input("Ask about your documents:")
        
        if user_question:
            with st.spinner("Analyzing..."):
                answer = user_input(user_question)
                if answer:
                    # Add new messages to the END of the list
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
            st.rerun()

    except Exception as e:
        logging.critical(f"App critical error: {str(e)}")
        st.error("Application encountered a critical error. Please refresh the page.")


if __name__ == "__main__":
    main()
