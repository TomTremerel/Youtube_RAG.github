import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import re

load_dotenv()

st.set_page_config(page_title="YT RAG", page_icon="📺")
st.title("📺Youtube Video RAG")

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([^&\n?]*)',
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/([^&\n?]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(url):
    """Get transcript from YouTube video."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return False
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        
        # Store transcript in session state instead of file
        st.session_state.transcript = text
        # Store the URL for video display
        st.session_state.video_url = url
        return True
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return False

def get_text_chunks(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_response(user_query, chat_history, vector_store):
    """Generate response using LLM."""
    try:
        template = """
        You are an expert on Video Youtube, you will have access to the transcript of the video provided and you will have to answer to the questions of user on this video. 
        
        Context: {context}
        Chat history: {chat_history}
        User question: {user_question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            model_name="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )

        # Get relevant documents from vector store
        docs = vector_store.similarity_search(user_query)
        context = "\n".join(doc.page_content for doc in docs)

        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "user_question": user_query,
        })
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am an Youtube expert. Please provide a YouTube video URL to begin."),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "video_url" not in st.session_state:
    st.session_state.video_url = None

# Sidebar for URL input and video display
with st.sidebar:
    st.title("Menu:")
    url = st.text_input(
        "Enter YouTube URL:",
        placeholder="e.g., https://www.youtube.com/watch?v=..."
    )
    
    if st.button("Process Video"):
        if url:
            with st.spinner("Fetching transcript..."):
                if get_transcript(url):
                    # Create vector store from transcript
                    chunks = get_text_chunks(st.session_state.transcript)
                    st.session_state.vector_store = create_vector_store(chunks)
                    if st.session_state.vector_store:
                        st.success("Video processed successfully!")
                    else:
                        st.error("Failed to process video.")
        else:
            st.warning("Please enter a YouTube URL.")
    
    # Display video if URL is available
    if st.session_state.video_url:
        st.video(st.session_state.video_url)

# Display chat history in main area
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Chat input
user_query = st.chat_input("Type your message here...")

if user_query:
    # Check if vector store is initialized
    if st.session_state.vector_store is None:
        st.warning("Please process a YouTube video first.")
    else:
        # Add user message to chat
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)

        # Generate and display AI response
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                response = get_response(
                    user_query,
                    st.session_state.chat_history,
                    st.session_state.vector_store
                )
                st.write(response)
                st.session_state.chat_history.append(AIMessage(content=response))