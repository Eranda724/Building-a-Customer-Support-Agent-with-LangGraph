# AI Customer Support Agent Builder

A powerful FastAPI application that allows businesses to create and customize their own AI-powered customer support agents. Built with LangGraph and modern AI technologies.

## 🌟 Features

- **Custom Agent Builder**: Create personalized support agents tailored to your business needs
- **Multi-step Configuration**: Guided process to set up your agent
- **Feature Suggestion System**: Smart suggestions based on your business type
- **Flexible Tone Settings**: Choose from professional, friendly, casual, or empathetic communication styles
- **Real-time Testing**: Test your custom agent immediately after creation

## 🛠️ Technology Stack

- **Backend**: FastAPI
- **AI Framework**: LangGraph
- **Language Model**: Groq
- **Frontend**: HTML/JavaScript (Single-page application)
- **State Management**: Custom state handling with LangGraph workflows

## 📋 Prerequisites

```bash
pip install fastapi uvicorn langgraph langchain-groq python-dotenv
```

## 🔧 Configuration

1. Create a .env file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies
3. Run the application:
```bash
uvicorn main:app --reload
```
4. Open http://localhost:8000 in your browser

## 💡 How to Use

### Creating a Custom Agent

1. Click "Start Building Custom Agent"
2. Enter your business name and description
3. Select or add custom features for your agent
4. Choose your agent's communication tone
5. Add any additional requirements
6. Save your agent and receive your unique Agent ID

### Testing Your Agent

1. Use the "Test Custom Agent" section
2. Enter your Agent ID
3. Type your test questions
4. Get real-time responses from your custom agent

## 🏗️ Project Structure

```
├── main.py                 # Main FastAPI application
├── custom_agent_engine.py  # Agent workflow and state management
├── agent_storage.py        # Agent configuration storage
└── requirements.txt        # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This is an active project under development. Features and documentation may be updated frequently.*
