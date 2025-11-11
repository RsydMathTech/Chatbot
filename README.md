# 💬 Gemini LLM Project by Rasyid

## Overview
This project demonstrates the integration of Gemini LLM with a data-driven chatbot interface. Users can upload tabular data and interact with it using natural language queries. The system translates user questions into SQL queries, executes them on the uploaded dataset, and returns clear, understandable answers.

## Goals
- Provide an **AI-powered assistant** for exploring and querying tabular data.  
- Enable **SQL query generation** via natural language using Gemini LLM.  
- Allow users to **upload multiple file formats**: CSV, Excel, JSON, Parquet.  
- Offer **schema inspection and sample data** previews automatically.  
- Maintain a **session-based conversation** to track ongoing queries.

## Key Features
- **Multi-format Data Upload:** Accepts CSV, Excel, JSON, and Parquet files.  
- **Natural Language to SQL:** Converts user questions into executable SQL queries.  
- **Interactive Chat:** Users can ask questions, receive answers, and iterate.  
- **Schema Overview:** Automatically generates table structure and sample previews.  
- **Customizable Model Parameters:** Supports temperature, top-p, top-k, and max token adjustments.  
- **Session Management:** Tracks conversation context and allows resetting the chat.  

## Usage
1. Upload your dataset in a supported file format.  
2. Select the desired Gemini model (e.g., `gemini-2.5-flash` or `gemini-2.5-pro`).  
3. Adjust model parameters if needed.  
4. Ask questions about your data in plain English.  
5. Review results displayed in a user-friendly format.  

## Benefits
- **Time-saving:** Quickly gain insights without manually writing SQL.  
- **User-friendly:** Non-technical users can interact with complex datasets.  
- **Flexible:** Works with a variety of data structures and formats.  
- **Transparent:** Provides schema information to help understand the data.  

## Project Repository
[GitHub - Chatbot by Rsyd](https://github.com/RsydMathTech/Chatbot)
