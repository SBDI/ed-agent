# Ed Agent: Advanced Educational AI Assistant

Ed Agent is a powerful educational AI assistant that combines:

- Personalized learning experiences tailored to various education levels
- Real-time information retrieval
- In-depth analysis and explanations

## Features
- 📚 Personalized education at various academic levels
- 🔍 Real-time information retrieval
- 📊 In-depth analysis and explanations
- 🧠 Interactive learning with quizzes and follow-up questions
- 💾 Save lessons for future reference

## Tech stack

- Llama 3.3 70B from Meta for the LLM
- Groq for LLM inference
- DuckDuckGo and Exa for the search API

## Cloning & running

> Note: Fork and clone this repository if needed

### 1. Create a virtual environment

```shell
uv venv
.venv\Scripts\activate
```

### 2. Install dependencies

```shell
uv pip install -r requirements.txt
```

### 3. Configure API Keys

Copy .env.example to .env and replace the API keys:
```bash
GROQ_API_KEY=your_groq_key_here
EXA_API_KEY=your_exa_key_here
```
### 4. Run

```shell
streamlit run app.py
```