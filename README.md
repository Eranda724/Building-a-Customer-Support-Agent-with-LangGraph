# AI Customer Support Agent Builder

A powerful FastAPI application that allows businesses to create and customize their own AI-powered customer support agents. Built with LangGraph and modern AI technologies.

## 🌟 Features

- **Custom Agent Builder**: Create personalized support agents tailored to your business needs
- **Multi-step Configuration**: Guided process to set up your agent
- **Feature Suggestion System**: Smart suggestions based on your business type
- **Flexible Tone Settings**: Choose from professional, friendly, casual, or empathetic communication styles
- **Real-time Testing**: Test your custom agent immediately after creation

<img width="1165" height="860" alt="1" src="https://github.com/user-attachments/assets/898263b9-d12d-4a97-8658-c43d35d044f6" />

![2](https://github.com/user-attachments/assets/7476bdce-6e4c-4f67-8f81-c1e6004e0493)
![3](https://github.com/user-attachments/assets/d8741b48-13ca-4c90-bcbb-efe3c4cb0cc3)
![4](https://github.com/user-attachments/assets/64daf43b-f9c4-43ad-9665-1f767aac9afe)
![5](https://github.com/user-attachments/assets/988b0e01-d77e-45d0-be9c-b734a6177019)
![6](https://github.com/user-attachments/assets/bf99bdff-2b50-46fd-9bd1-73515e5fb25c)
![7](https://github.com/user-attachments/assets/49fc2684-34e4-4d0b-98b6-d8ae47aa04ea)

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
