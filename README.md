# Universal AI Customer Support Agent Builder

A powerful FastAPI application that allows users to create and customize AI-powered customer support agents for ANY type of business, system, task, or job. Built with LangGraph and modern AI technologies.

## 🌟 Features

- **Universal Agent Creation**: Create support agents for websites, companies, systems, tasks, or any work scenario
- **🤖 CustomGPT Guide Generator**: AI-powered guidance for creating agents - just describe what you need!
- **User Prompt Storage**: Save and manage your prompts as individual JSON databases
- **Session Management**: Create user sessions to organize multiple prompts
- **Dynamic Business Types**: Automatically adapts to any business context using universal prompts
- **Custom Agent Builder**: Create personalized support agents tailored to your specific needs
- **Multi-step Configuration**: Guided process to set up your agent
- **Flexible Tone Settings**: Choose from professional, friendly, casual, or empathetic communication styles
- **Food Delivery Specialization**: Built-in support for restaurant and food delivery businesses with menu/pricing integration
- **Menu Data Extraction**: Automatically extracts menu items, prices, and specials from business descriptions
- **Voice Integration**: Support for voice-based interactions (optional)
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
- **Language Model**: Grog
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

### 🤖 CustomGPT Guide Generator

Get AI-powered guidance for creating your customer support agents:

1. **Ask for Guidance**: Describe what kind of agent you want to create
2. **Get Complete Guide**: Receive business name, description, features, and tone suggestions
3. **One-Click Application**: Apply the generated guide directly to your agent builder

**Example Queries:**

- "Give me a guide for a food delivery website agent with menu, pricing, and order features"
- "Create a guide for an e-commerce store selling electronics"
- "Help me build an agent for a healthcare clinic with appointment booking"
- "Guide for a banking app with account management and support"

### Universal Agent Creation

Create agents for ANY type of business, system, or task:

1. **Create a User Session**: Start by creating a user session to organize your prompts
2. **Describe Your System**: Provide detailed information about your website, company, system, or task
3. **Save Your Prompt**: Each prompt is saved as a separate JSON file for future use
4. **Load Previous Prompts**: Access and reuse your saved prompts anytime

**Examples:**

- **Website Support**: "My e-commerce website sells electronics. We offer laptops, phones, and accessories with free shipping over $50"
- **Company HR**: "Tech startup providing software development services. We hire developers, designers, and project managers"
- **Task Management**: "Project management tool for teams. Features include task assignment, deadline tracking, and progress reporting"
- **Service Business**: "Plumbing service company. We handle residential and commercial plumbing repairs, installations, and maintenance"

### Food Delivery Agent Setup

For food delivery businesses, include menu items, prices, and specials in your business description:

```
Example: "Mario's Pizza - We serve authentic Italian pizza with prices from $12-25.
Menu includes: Margherita Pizza $15, Pepperoni Pizza $18, Delivery fee $3, Free delivery over $30"
```

The system will automatically extract:

- Menu items and prices
- Delivery fees
- Special offers and promotions

### Voice Integration (Optional)

The system includes voice interaction capabilities:

- Voice query endpoints for speech-to-text
- Audio response generation
- Integration with ElevenLabs for text-to-speech

Enable voice features by setting up the required API keys in your .env file.

## 🏗️ Project Structure

```
├── main.py                 # Main entry point
├── src/
│   ├── __init__.py
│   ├── main.py             # Main FastAPI application
│   ├── custom_agent_engine.py  # Agent workflow and state management
│   └── agent_storage.py    # Agent configuration storage
├── prompts/
│   └── default_prompts.json # Default prompts for different business types
├── agents/
│   └── configs/            # Stored agent configurations
├── voice/                  # Voice interaction features (optional)
├── requirements.txt        # Project dependencies
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

_Note: This is an active project under development. Features and documentation may be updated frequently._
