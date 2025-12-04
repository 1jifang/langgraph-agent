# 🤖 LangGraph ReAct Agent

> A smart AI assistant that can "think", use tools, and remember your conversation. Built with **LangGraph** and **Nebius AI**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/AI-LangGraph-orange)
![Nebius](https://img.shields.io/badge/Model-Qwen%2FLlama-green)

## 📖 Introduction 

This is an AI Agent project that goes beyond a simple chatbot.

Most chatbots just talk, but this Agent can **take action**. It uses the **ReAct (Reasoning + Acting)** architecture to:
1.  **Check real-time weather** ☀️ (using Open-Meteo API).
2.  **Search Wikipedia** 📚 (to answer general knowledge questions).
3.  **Remember context** 🧠 (it knows what you said previously).

It runs in your terminal with a cool "Cyberpunk" style interface! 

## ✨ Features

* **Smart Reasoning**: The AI decides when to use tools and when to just chat.
* **Memory**: You can follow up with questions like "What about Shanghai?" and it knows you are still talking about weather.
* **Cool UI**: Powered by `rich`, showing you the AI's thought process in a beautiful way.
* **Free APIs**: Uses Nebius AI (LLM) and Open-Meteo (Weather), no credit card needed for the weather part.

---

## 🚀 How to Start 

Follow these simple steps to run this bot on your computer.

### 1. Prerequisites 
Make sure you have **Python 3.10** or higher installed.

### 2. Clone the Repository 
Open your terminal (CMD or PowerShell) and run: git clone [https://github.com/1jifang/langgraph-agent-demo.git](https://github.com/1jifang/langgraph-agent-demo.git)
cd langgraph-agent-demo


### 3. Install Dependencies
Run this in your terminal: pip install langgraph langchain-litellm langchain-community langchain-core litellm geopy requests rich wikipedia pydantic tenacity

### 4. Set up API Key
You need an API Key from Nebius AI Studio.
Get your key from Nebius Studio. And Find the line os.environ["NEBIUS_API_KEY"] and paste your key

### 5. Run it!
Start the agent: python advanced_agent.py

### How to Use:
Once the agent is running, you will see a green prompt You >. Just type your questions!

Try these examples:

Check Weather:

"What is the weather in Berlin right now?"

Multi-step Reasoning:

"How is the weather in Tokyo and tell me a fun fact about the city."

Memory Check:

(First ask about Berlin weather) -> Then ask: "Is it warmer than London?"

General Chat:

"Write a hello world function in Python."
