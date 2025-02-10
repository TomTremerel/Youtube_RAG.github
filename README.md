# Youtube transcript RAG

## Overview
This is a project that transforms YouTube videos into interactive learning experiences by enabling context-aware conversations about video content. By leveraging video transcripts and RAG (Retrieval-Augmented Generation) technology, users can engage in meaningful discussions about any YouTube video's content.

## How It Works

### 1. Transcript Extraction
- Utilizes `YoutubeTranscriptApi` to automatically extract text transcripts from any YouTube video
- Processes and cleanses the transcript data for optimal analysis

### 2. RAG Implementation
- Splits transcript text into manageable chunks for processing
- Creates embeddings and stores them in a vector database
- Enables semantic search and contextual retrieval of video content

### 3. Interactive Chat Interface
- Watch the video while chatting with the AI assistant
- Ask questions about specific parts of the video
- Get context-aware responses based on the video's content

## Architecture
![RAG Architecture](https://github.com/user-attachments/assets/7ab323ad-650d-4213-bacf-3c89faebc627)

## Current Limitations
The system currently relies solely on video transcripts, meaning it only understands what is verbally expressed in the video. This leads to some limitations:
- Cannot interpret visual content or demonstrations
- Misses non-verbal communication
- Unable to understand visual examples or diagrams

## Future Improvements
The project could be significantly enhanced by implementing multimodal AI capabilities:
- Visual content analysis
- Scene understanding
- Gesture recognition
- Integration of both audio and visual context
- Real-time video frame analysis

This would provide a more comprehensive understanding of the video content, allowing for richer and more accurate interactions with users.
