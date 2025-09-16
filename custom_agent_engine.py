from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr
from langgraph.graph import StateGraph, END
import re
import os

class State(BaseModel):
    query: str
    category: str = ""
    sentiment: str = ""
    response: str = ""

def extract_content(response: Any) -> str:
    """Extract and clean content from LLM response"""
    if isinstance(response, AIMessage):
        content = str(response.content)
    elif isinstance(response, dict) and "content" in response:
        content = str(response["content"])
    else:
        content = str(response)
    
    return clean_markdown_formatting(content)

def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting from text"""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^[-*]{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()

def build_agent_workflow(config: Dict[str, Any], llm: Optional[ChatGroq] = None) -> Any:
    """Build a reusable agent workflow from configuration"""
    
    if llm is None:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        llm = ChatGroq(
            temperature=0.7,
            api_key=api_key,
            model="moonshotai/kimi-k2-instruct"
        )

    prompts = config.get("prompts", {})
    features = config.get("features", [])
    tone = config.get("tone", "professional")

    def categorize(state: State) -> State:
        """Categorize incoming queries"""
        query_lower = state.query.lower()

        # Smart categorization based on keywords
        if any(word in query_lower for word in ['menu', 'food', 'rice', 'kottu', 'eat', 'dish', 'price', 'cost']):
            state.category = "information"
            return state
        elif any(word in query_lower for word in ['open', 'close', 'hour', 'time', 'available', 'when', 'monday', 'friday', 'saturday']):
            state.category = "information"
            return state
        elif any(word in query_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'greetings']):
            state.category = "general"
            return state
        elif any(word in query_lower for word in ['help', 'support', 'problem', 'issue', 'error']):
            state.category = "support"
            return state
        else:
            # Use LLM for complex queries
            prompt_template = prompts.get("categorize",
                "Categorize this customer query into: Information, Support, Issues, General. Query: {query}")

            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            response = chain.invoke({"query": state.query})
            state.category = extract_content(response).strip().lower()
            return state

    def analyze_sentiment(state: State) -> State:
        """Analyze query sentiment"""
        prompt_template = prompts.get("sentiment",
            "Analyze sentiment as positive, negative, or neutral. Query: {query}")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({"query": state.query})
        state.sentiment = extract_content(response).strip().lower()
        return state

    def handle_query(state: State) -> State:
        """Handle queries based on category and features"""
        category = state.category
        query_lower = state.query.lower()

        # Determine if query is about specific topics
        is_menu_query = any(word in query_lower for word in ['menu', 'food', 'rice', 'kottu', 'eat', 'dish', 'price', 'cost'])
        is_availability_query = any(word in query_lower for word in ['open', 'close', 'hour', 'time', 'available', 'when', 'monday', 'friday', 'saturday', 'vacation'])
        is_greeting = any(word in query_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'greetings'])

        prompt_key = f"handle_{category}" if category in ["technical", "billing", "menu", "pricing", "order", "delivery"] else "general"

        # Get category-specific prompt or fallback to general
        prompt_template = prompts.get(prompt_key, prompts.get("general",
            "Provide a {tone} response to: {query}"))

        # Only add relevant custom information based on query type
        custom_features = config.get("custom_features", [])
        relevant_context = ""

        if is_menu_query and custom_features:
            # Find menu-related custom features
            menu_features = [f for f in custom_features if 'menu' in f['name'].lower()]
            if menu_features:
                relevant_context = "Menu Information:\n" + "\n".join(f"- {f['description']}" for f in menu_features)

        elif is_availability_query and custom_features:
            # Find availability-related custom features
            avail_features = [f for f in custom_features if 'available' in f['name'].lower()]
            if avail_features:
                relevant_context = "Availability Information:\n" + "\n".join(f"- {f['description']}" for f in avail_features)

        # For greetings, don't add any custom context
        if is_greeting:
            relevant_context = ""

        # Add relevant context to prompt if available
        if relevant_context:
            prompt_template = f"{relevant_context}\n\n{prompt_template}"

        # Add feature context only for complex queries
        if not is_greeting and features:
            feature_context = "\n".join(f"- {f['name']}: {f['description']}" for f in features)
            prompt_template = f"Available features:\n{feature_context}\n\n{prompt_template}"

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({
            "query": state.query,
            "tone": tone
        })
        state.response = extract_content(response)
        return state

    def route_query(state: State) -> str:
        """Route queries based on sentiment and category"""
        if state.sentiment == "negative":
            return "escalate"
        return "handle_query"

    def escalate(state: State) -> State:
        """Handle escalation for negative sentiment"""
        state.response = prompts.get("escalate",
            "This query requires attention from our support team. A representative will contact you shortly.")
        return state

    # Build the workflow graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("categorize", categorize)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("handle_query", handle_query)
    workflow.add_node("escalate", escalate)

    # Add edges
    workflow.add_edge("categorize", "analyze_sentiment")
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_query,
        {
            "handle_query": "handle_query",
            "escalate": "escalate"
        }
    )
    workflow.add_edge("handle_query", END)
    workflow.add_edge("escalate", END)

    workflow.set_entry_point("categorize")
    
    return workflow.compile() 