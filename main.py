from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr, validator
from typing import Dict, Any, Union, cast, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
import json
import uuid
from enum import Enum
from datetime import datetime
import logging
from functools import wraps
import re

class FeaturePriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Feature(BaseModel):
    id: str
    name: str
    description: str
    priority: FeaturePriority
    is_custom: bool = False

    @validator('priority', pre=True)
    def validate_priority(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in FeaturePriority.__members__:
                return FeaturePriority[value]
        elif isinstance(value, FeaturePriority):
            return value
        raise ValueError(f"Priority must be one of {list(FeaturePriority.__members__.keys())}")

class State(BaseModel):
    query: str
    category: str = ""
    sentiment: str = ""
    response: str = ""

class AgentBuilderState(BaseModel):
    session_id: str
    current_step: int = 0
    business_name: str = ""
    business_description: str = ""
    business_type: str = ""
    business_context: str = ""
    key_terms: List[str] = []
    interaction_points: List[str] = []
    selected_features: List[Feature] = Field(default_factory=list)
    suggested_features: List[Feature] = Field(default_factory=list)
    custom_features: List[Feature] = Field(default_factory=list)
    tone: str = ""
    custom_prompts: Dict[str, str] = Field(default_factory=dict)
    generated_config: Dict[str, Any] = Field(default_factory=dict)
    is_complete: bool = False
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BusinessType(str, Enum):
    FOOD_DELIVERY = "food_delivery"
    BOOKING_SYSTEM = "booking_system"
    COMMUNICATION_PROVIDER = "communication_provider"
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    CUSTOM = "custom"

class AgentFeature(str, Enum):
    ORDER_TRACKING = "order_tracking"
    PAYMENT_SUPPORT = "payment_support"
    BOOKING_MANAGEMENT = "booking_management"
    TECHNICAL_SUPPORT = "technical_support"
    PRODUCT_RECOMMENDATIONS = "product_recommendations"
    APPOINTMENT_SCHEDULING = "appointment_scheduling"
    REFUND_PROCESSING = "refund_processing"
    MULTILINGUAL_SUPPORT = "multilingual_support"

# Initialize LLM
GROQ_API_KEY = SecretStr("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")

llm = ChatGroq(
    temperature=0.7,
    api_key=SecretStr(GROQ_API_KEY.get_secret_value()),
    model="mistral-saba-24b"
)

# Store agent builder sessions in memory (use Redis/Database in production)
agent_builder_sessions: Dict[str, AgentBuilderState] = {}
custom_agents: Dict[str, Dict[str, Any]] = {}

def extract_content(response: Any) -> str:
    if isinstance(response, AIMessage):
        content = str(response.content)
    elif isinstance(response, dict) and "content" in response:
        content = str(response["content"])
    elif isinstance(response, (str, list)):
        content = str(response)
    else:
        content = str(response)
    
    # Clean markdown formatting
    return clean_markdown_formatting(content)

def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting and convert to plain text"""
    
    # Remove bold formatting (**text** -> text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove italic formatting (*text* -> text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove code formatting (`text` -> text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove headers (# Header -> Header)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove horizontal rules (--- or ***)
    text = re.sub(r'^[-*]{3,}$', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

# Original agent functions
def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of the following categories: "
        "Technical, Billing, General. Query: {query}."
    )
    chain = prompt | llm
    response = chain.invoke({"query": state.query})
    state.category = extract_content(response).strip().lower()
    return state

def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either positive, negative, or neutral. Query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state.query})
    state.sentiment = extract_content(response).strip().lower()
    return state

def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a clear, step-by-step technical support response to the following query. "
        "Use simple, plain text formatting with clear sections and easy-to-follow instructions. "
        "Avoid markdown formatting like #, *, or **. Just use plain text with proper spacing. "
        "Make the instructions easy to follow and well-spaced. Query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state.query})
    state.response = extract_content(response)
    return state

def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a clear, well-structured response to the following billing query. "
        "Use simple, plain text formatting with clear sections and bullet points where needed. "
        "Avoid markdown formatting like #, *, or **. Just use plain text with proper spacing. "
        "Keep sensitive information private and be clear about policies. Query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state.query})
    state.response = extract_content(response)
    return state

def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a clear, well-formatted response to the following customer query. "
        "Use simple, plain text formatting with proper spacing and clear sections. "
        "Avoid markdown formatting like #, *, or **. Just use plain text with proper spacing. "
        "Keep the tone professional and friendly. Query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state.query})
    state.response = extract_content(response)
    return state

def escalate(state: State) -> State:
    state.response = "This query has been escalated to a human agent due to its negative sentiment."
    return state

def route_query(state: State) -> str:
    if state.sentiment == "negative":
        return "escalate"
    elif state.category == "billing":
        return "handle_billing"
    elif state.category == "technical":
        return "handle_technical"
    else:
        return "handle_general"

def analyze_business_context(business_name: str, business_description: str = "") -> Dict[str, Any]:
    """Intelligently analyze business context using LLM"""
    
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following business name and description to understand its context and suggest relevant customer support features.
    
    Business Name: {business_name}
    Business Description: {business_description}
    
    Please provide a JSON response with the following structure:
    {
        "business_category": "primary business category",
        "business_context": "2-3 sentence description of what the business does",
        "suggested_features": [
            {
                "id": "feature_id_in_snake_case",
                "name": "Feature Name",
                "description": "what this feature does",
                "priority": "high/medium/low"
            }
        ],
        "key_terms": ["relevant", "industry", "specific", "terms"],
        "customer_interaction_points": ["where", "customers", "typically", "need", "support"],
        "recommended_tone": "professional/friendly/casual/empathetic"
    }
    
    Focus on understanding the unique aspects of this business and suggest features that would be most valuable for their customer support needs.
    """)
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "business_name": business_name,
            "business_description": business_description
        })

        # Parse the JSON response
        result = json.loads(extract_content(response))

        # Validate the result structure
        if not all(key in result for key in ["business_category", "business_context", "suggested_features", "key_terms", "customer_interaction_points", "recommended_tone"]):
            raise ValueError("Invalid response structure from LLM")

        # Convert suggested features to Feature objects
        suggested_features = [
            Feature(
                id=feature.get("id", ""),
                name=feature.get("name", ""),
                description=feature.get("description", ""),
                priority=FeaturePriority[feature.get("priority", "MEDIUM").upper()]
            )
            for feature in result.get("suggested_features", [])
        ]

        return {
            "business_type": result["business_category"],
            "business_context": result["business_context"],
            "suggested_features": suggested_features,
            "key_terms": result["key_terms"],
            "interaction_points": result["customer_interaction_points"],
            "recommended_tone": result["recommended_tone"]
        }
    except json.JSONDecodeError as e:
        # Handle JSON decode error specifically
        return {
            "business_type": "custom",
            "business_context": f"Business providing {business_name} services",
            "suggested_features": [
                Feature(
                    id="general_support",
                    name="General Support",
                    description="Basic customer support capabilities",
                    priority=FeaturePriority.HIGH
                )
            ],
            "key_terms": [business_name.lower()],
            "interaction_points": ["general inquiries"],
            "recommended_tone": "professional"
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error analyzing business context: {str(e)}")
        return {
            "business_type": "custom",
            "business_context": f"Business providing {business_name} services",
            "suggested_features": [
                Feature(
                    id="general_support",
                    name="General Support",
                    description="Basic customer support capabilities",
                    priority=FeaturePriority.HIGH
                )
            ],
            "key_terms": [business_name.lower()],
            "interaction_points": ["general inquiries"],
            "recommended_tone": "professional"
        }

# Custom agent builder functions
def detect_business_type(business_name: str) -> tuple[str, List[str]]:
    """Intelligently detect business type and suggest features based on business name"""
    
    business_keywords = {
        "food_delivery": {
            "keywords": ["pizza", "food", "delivery", "restaurant", "cafe", "kitchen", "burger", "sushi", "bakery", "catering", "dine", "eat", "meal", "takeaway", "takeout"],
            "suggested_features": ["order_tracking", "payment_support", "refund_processing", "delivery_updates", "menu_inquiries", "allergy_information"]
        },
        "booking_system": {
            "keywords": ["hotel", "booking", "reservation", "appointment", "schedule", "salon", "spa", "clinic", "dental", "medical", "therapy", "consulting", "meeting"],
            "suggested_features": ["appointment_scheduling", "booking_management", "calendar_integration", "reminder_notifications", "cancellation_policy", "availability_check"]
        },
        "communication_provider": {
            "keywords": ["telecom", "mobile", "internet", "phone", "network", "broadband", "wifi", "cellular", "communication", "provider", "isp"],
            "suggested_features": ["technical_support", "billing_support", "plan_information", "network_troubleshooting", "account_management", "service_upgrades"]
        },
        "ecommerce": {
            "keywords": ["shop", "store", "retail", "marketplace", "boutique", "fashion", "electronics", "gadgets", "online", "commerce", "trade", "market"],
            "suggested_features": ["order_tracking", "payment_support", "product_recommendations", "refund_processing", "inventory_inquiries", "shipping_information"]
        },
        "healthcare": {
            "keywords": ["health", "medical", "hospital", "pharmacy", "doctor", "nurse", "patient", "treatment", "medicine", "wellness", "care", "clinic"],
            "suggested_features": ["appointment_scheduling", "medical_inquiries", "prescription_support", "insurance_verification", "health_records", "emergency_assistance"]
        },
        "education": {
            "keywords": ["school", "university", "college", "education", "learning", "course", "training", "academy", "institute", "study", "student", "teacher"],
            "suggested_features": ["course_information", "enrollment_support", "academic_assistance", "schedule_management", "payment_support", "technical_support"]
        },
        "travel": {
            "keywords": ["travel", "tour", "vacation", "flight", "airline", "cruise", "trip", "adventure", "holiday", "tourism", "visa", "passport"],
            "suggested_features": ["booking_management", "itinerary_support", "travel_insurance", "visa_assistance", "flight_updates", "cancellation_support"]
        },
        "finance": {
            "keywords": ["bank", "finance", "loan", "credit", "investment", "insurance", "financial", "money", "payment", "account", "mortgage"],
            "suggested_features": ["account_management", "transaction_support", "loan_inquiries", "investment_advice", "security_assistance", "billing_support"]
        },
        "automotive": {
            "keywords": ["car", "auto", "vehicle", "garage", "mechanic", "repair", "service", "dealership", "automotive", "motor", "truck"],
            "suggested_features": ["service_scheduling", "repair_status", "parts_availability", "warranty_information", "maintenance_reminders", "technical_support"]
        },
        "real_estate": {
            "keywords": ["property", "real estate", "house", "apartment", "rent", "lease", "mortgage", "realtor", "home", "building", "land"],
            "suggested_features": ["property_inquiries", "viewing_appointments", "application_support", "maintenance_requests", "payment_processing", "lease_information"]
        }
    }
    
    business_name_lower = business_name.lower()
    detected_type = "custom"
    suggested_features = []
    
    # Find matching business type
    for biz_type, data in business_keywords.items():
        for keyword in data["keywords"]:
            if keyword in business_name_lower:
                detected_type = biz_type
                suggested_features = data["suggested_features"]
                break
        if detected_type != "custom":
            break
    
    return detected_type, suggested_features

def get_business_feature_suggestions(business_type: str, existing_features: List[str] = []) -> List[str]:
    """Get initial feature suggestions based on business type"""
    
    all_features = {
        "order_tracking": "Track orders and delivery status",
        "payment_support": "Handle payments, refunds, and billing",
        "booking_management": "Manage appointments and reservations", 
        "technical_support": "Resolve technical issues and troubleshooting",
        "product_recommendations": "Suggest products based on customer needs",
        "appointment_scheduling": "Schedule and manage appointments",
        "refund_processing": "Process refunds and handle returns",
        "multilingual_support": "Support multiple languages",
        "delivery_updates": "Real-time delivery notifications",
        "menu_inquiries": "Answer questions about menu items",
        "allergy_information": "Provide allergen and dietary information",
        "calendar_integration": "Sync with calendar systems",
        "reminder_notifications": "Send appointment reminders",
        "cancellation_policy": "Explain cancellation terms",
        "availability_check": "Check real-time availability",
        "plan_information": "Provide service plan details",
        "network_troubleshooting": "Diagnose network issues",
        "account_management": "Manage customer accounts",
        "service_upgrades": "Handle service upgrade requests",
        "inventory_inquiries": "Check product availability",
        "shipping_information": "Provide shipping details",
        "medical_inquiries": "Answer basic medical questions",
        "prescription_support": "Help with prescription queries",
        "insurance_verification": "Verify insurance coverage",
        "health_records": "Access health record information",
        "emergency_assistance": "Provide emergency support",
        "course_information": "Provide course details",
        "enrollment_support": "Help with enrollment process",
        "academic_assistance": "Provide academic support",
        "schedule_management": "Manage class schedules",
        "itinerary_support": "Help with travel itineraries",
        "travel_insurance": "Provide travel insurance info",
        "visa_assistance": "Help with visa requirements",
        "flight_updates": "Provide flight status updates",
        "transaction_support": "Help with financial transactions",
        "loan_inquiries": "Answer loan-related questions",
        "investment_advice": "Provide investment guidance",
        "security_assistance": "Help with security concerns",
        "service_scheduling": "Schedule service appointments",
        "repair_status": "Check repair progress",
        "parts_availability": "Check parts availability",
        "warranty_information": "Provide warranty details",
        "maintenance_reminders": "Send maintenance alerts",
        "property_inquiries": "Answer property questions",
        "viewing_appointments": "Schedule property viewings",
        "application_support": "Help with applications",
        "maintenance_requests": "Handle maintenance requests",
        "lease_information": "Provide lease details"
    }
    
    # Filter out already selected features
    available_features = {k: v for k, v in all_features.items() if k not in existing_features}
    
    return list(available_features.keys())[:12]  # Return top 12 suggestions

def generate_agent_prompts(
    business_type: str,
    features: List[Feature],
    tone: str,
    custom_requirements: str = "",
    business_name: str = "your business"
) -> Dict[str, str]:
    """Generate customized prompts based on user requirements"""

    # Group features by priority
    high_priority_features = [f"{feature.name}: {feature.description}" for feature in features if feature.priority == FeaturePriority.HIGH]
    medium_priority_features = [f"{feature.name}: {feature.description}" for feature in features if feature.priority == FeaturePriority.MEDIUM]
    low_priority_features = [f"{feature.name}: {feature.description}" for feature in features if feature.priority == FeaturePriority.LOW]

    # Build the features section with priority grouping
    features_section = ""
    if high_priority_features:
        features_section += "High Priority Capabilities:\n" + "\n".join(f"- {feature}" for feature in high_priority_features) + "\n\n"
    if medium_priority_features:
        features_section += "Medium Priority Capabilities:\n" + "\n".join(f"- {feature}" for feature in medium_priority_features) + "\n\n"
    if low_priority_features:
        features_section += "Low Priority Capabilities:\n" + "\n".join(f"- {feature}" for feature in low_priority_features) + "\n\n"

    # Add additional context based on business type
    business_context = ""
    if business_type == BusinessType.ECOMMERCE:
        business_context = "As an e-commerce business, focus on order tracking, product information, and payment issues."
    elif business_type == BusinessType.HEALTHCARE:
        business_context = "As a healthcare provider, prioritize patient privacy, appointment scheduling, and medical information accuracy."
    # Add more business type specific contexts as needed

    base_prompt = f"""
    You are a professional customer support agent for {business_name}, a {business_type} business.
    {business_context}
    Always maintain a {tone} tone in your responses.
    
    IMPORTANT: Use only plain text formatting. Do not use markdown symbols like #, *, or **. 
    Use simple text with proper spacing, clear sections, and easy-to-read formatting.

    {features_section}

    Additional guidelines:
    - Always greet the customer politely
    - Be concise but thorough in your responses
    - If you don't know an answer, offer to connect the customer with a specialist
    - {custom_requirements}

    Your goal is to provide excellent customer support and resolve issues efficiently for {business_name}.
    """

    return {
        "categorize": f"{base_prompt}\n\nCategorize this customer query into the most appropriate category for {business_name}.\n"
                     f"Available categories: Technical, Billing, General, or Other if it doesn't fit.\n"
                     f"Respond with ONLY the category name.\n"
                     f"Query: {{query}}",
        "technical": f"{base_prompt}\n\nProvide detailed technical support for this {business_name} query.\n"
                    f"Be sure to:\n"
                    f"- Acknowledge the customer's issue\n"
                    f"- Provide step-by-step guidance if applicable\n"
                    f"- Offer additional help if the issue isn't resolved\n"
                    f"Use plain text formatting only - no markdown symbols.\n"
                    f"Query: {{query}}",
        "billing": f"{base_prompt}\n\nHandle this billing/payment related query for {business_name}.\n"
                  f"Important notes:\n"
                  f"- Never share sensitive customer information\n"
                  f"- Be clear about payment policies\n"
                  f"- Offer to connect with accounting for complex issues\n"
                  f"Use plain text formatting only - no markdown symbols.\n"
                  f"Query: {{query}}",
        "general": f"{base_prompt}\n\nProvide comprehensive general customer support for this {business_name} query.\n"
                  f"Remember to:\n"
                  f"- Be friendly and welcoming\n"
                  f"- Provide complete information\n"
                  f"- Offer additional assistance\n"
                  f"Use plain text formatting only - no markdown symbols.\n"
                  f"Query: {{query}}",
        "escalate": f"{base_prompt}\n\nThis query needs to be escalated to a human agent.\n"
                   f"Politely inform the customer that their issue will be handled by a specialist.\n"
                   f"Provide an estimated wait time if possible.\n"
                   f"Use plain text formatting only - no markdown symbols.\n"
                   f"Query: {{query}}"
    }


def create_custom_agent(config: Dict[str, Any]) -> Any:
    """Create a custom agent based on configuration with enhanced error handling and logging"""

    # Validate configuration
    if not config or "prompts" not in config:
        raise ValueError("Invalid agent configuration: missing prompts")

    prompts = config.get("prompts", {})

    # Create a logger for the agent
    logger = logging.getLogger(f"custom_agent_{uuid.uuid4()}")

    def log_and_continue(func):
        """Decorator to log function execution and handle errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Executing {func.__name__} with query: {args[0].query}")
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                # Return a state with an error response
                state = args[0]
                state.response = f"I'm sorry, I encountered an error while processing your request. Please try again later."
                return state
        return wrapper

    @log_and_continue
    def custom_categorize(state: State) -> State:
        prompt_template = prompts.get("categorize", "Categorize this customer query: {query}")
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            response = chain.invoke({"query": state.query})
            content = extract_content(response)
            state.category = content.strip().lower()
            logger.info(f"Categorized query as: {state.category}")
        except Exception as e:
            logger.error(f"Error in categorization: {str(e)}")
            state.category = "general"
        return state

    @log_and_continue
    def custom_technical(state: State) -> State:
        prompt_template = prompts.get("technical", "Provide technical support for this query: {query}")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({"query": state.query})
        state.response = extract_content(response)
        return state

    @log_and_continue
    def custom_billing(state: State) -> State:
        prompt_template = prompts.get("billing", "Provide billing support for this query: {query}")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({"query": state.query})
        state.response = extract_content(response)
        return state

    @log_and_continue
    def custom_general(state: State) -> State:
        prompt_template = prompts.get("general", "Provide general support for this query: {query}")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({"query": state.query})
        state.response = extract_content(response)
        return state

    @log_and_continue
    def custom_escalate(state: State) -> State:
        try:
            # Try to use a custom escalation prompt if available
            if "escalate" in prompts:
                prompt_template = prompts.get("escalate")
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | llm
                response = chain.invoke({"query": state.query})
                state.response = extract_content(response)
            else:
                state.response = "I'm sorry, I need to escalate this issue to a human agent. Please hold on while I transfer you."
        except Exception as e:
            logger.error(f"Error in escalation: {str(e)}")
            state.response = "I'm sorry, I need to escalate this issue to a human agent. Please hold on while I transfer you."
        return state

    # Create workflow with enhanced error handling
    workflow = StateGraph(State)

    # Add nodes with error handling
    try:
        workflow.add_node("categorize", custom_categorize)
        workflow.add_node("analyze_sentiment", analyze_sentiment)
        workflow.add_node("handle_billing", custom_billing)
        workflow.add_node("handle_technical", custom_technical)
        workflow.add_node("handle_general", custom_general)
        workflow.add_node("escalate", custom_escalate)

        # Add edges
        workflow.add_edge("categorize", "analyze_sentiment")

        # Enhanced routing with fallback
        def enhanced_route_query(state: State) -> str:
            try:
                if state.sentiment == "negative":
                    return "escalate"
                elif state.category == "billing":
                    return "handle_billing"
                elif state.category == "technical":
                    return "handle_technical"
                else:
                    return "handle_general"
            except Exception as e:
                logger.error(f"Error in routing: {str(e)}")
                return "handle_general"

        workflow.add_conditional_edges(
            "analyze_sentiment",
            enhanced_route_query,
            {
                "handle_technical": "handle_technical",
                "handle_billing": "handle_billing",
                "handle_general": "handle_general",
                "escalate": "escalate"
            }
        )

        # Add final edges
        workflow.add_edge("escalate", END)
        workflow.add_edge("handle_technical", END)
        workflow.add_edge("handle_billing", END)
        workflow.add_edge("handle_general", END)

        workflow.set_entry_point("categorize")

        # Compile and return the workflow
        return workflow.compile()

    except Exception as e:
        logger.error(f"Error creating workflow: {str(e)}")
        raise


# Original workflow
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query, {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)

workflow.add_edge("escalate", END)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)

workflow.set_entry_point("categorize")
chain = workflow.compile()

app = FastAPI(
    title="Customer Support Agent API with Custom Agent Builder",
    description="An AI-powered customer support system with custom agent creation capabilities.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class SupportResponse(BaseModel):
    category: str
    sentiment: str
    response: str

class StartAgentBuilderRequest(BaseModel):
    pass

class AgentBuilderResponse(BaseModel):
    session_id: str
    step: int
    question: str
    options: Optional[List[str]] = None
    is_complete: bool = False
    agent_id: Optional[str] = None
    business_type_info: Optional[str] = None
    suggestions: Optional[List[str]] = None

class AgentBuilderStepRequest(BaseModel):
    session_id: str
    answer: str

class CustomAgentQuery(BaseModel):
    agent_id: str
    query: str

# Request models for feature management
class CustomFeatureRequest(BaseModel):
    session_id: str
    feature_name: str
    feature_description: str = "Custom feature added by user"

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Support Agent Builder</title>
        <style>
            :root {
                --primary-color: #2563eb;
                --secondary-color: #1e40af;
                --background-color: #f8fafc;
                --text-color: #1e293b;
                --border-color: #e2e8f0;
                --success-color: #059669;
                --container-width: 900px;
            }

            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: var(--background-color);
                color: var(--text-color);
            }

            .header {
                background-color: white;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                position: fixed;
                width: 100%;
                top: 0;
                z-index: 100;
            }

            .header-content {
                max-width: var(--container-width);
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .logo {
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--primary-color);
            }

            .get-help-btn {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-weight: 600;
                transition: background-color 0.2s;
            }

            .get-help-btn:hover {
                background-color: var(--secondary-color);
            }

            .main-container {
                max-width: var(--container-width);
                margin: 6rem auto 2rem;
                padding: 0 1rem;
                transition: max-width 0.3s;
            }

            .main-container.fullscreen {
                max-width: 100%;
                margin: 6rem 2rem 2rem;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
            }

            .container {
                background-color: white;
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 2rem;
            }

            .container h2 {
                margin-top: 0;
                color: var(--primary-color);
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }

            .container p {
                color: #64748b;
                margin-bottom: 1.5rem;
            }

            textarea, input[type="text"] {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                font-family: inherit;
            }

            button {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-weight: 600;
                margin: 0.5rem;
                transition: background-color 0.2s;
            }

            button:hover {
                background-color: var(--secondary-color);
            }

            .option-btn {
                background-color: #f1f5f9;
                color: var(--text-color);
            }

            .option-btn:hover {
                background-color: #e2e8f0;
            }

            .response-box {
                background-color: #f8fafc;
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                padding: 1rem;
                margin-top: 1rem;
                white-space: pre-wrap;
                font-family: inherit;
                line-height: 1.6;
            }

            .loading {
                display: none;
                color: var(--primary-color);
                margin: 1rem 0;
                font-weight: 500;
            }

            .feature-input-container {
                display: flex;
                gap: 1rem;
                margin: 1rem 0;
            }

            .feature-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin: 1rem 0;
            }

            .feature-tag {
                background-color: var(--primary-color);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 1rem;
                font-size: 0.875rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .feature-tag .remove-btn {
                background: none;
                border: none;
                color: white;
                padding: 0;
                margin: 0;
                cursor: pointer;
                font-weight: bold;
            }

            .suggestions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }

            .suggestion-btn {
                text-align: left;
                background-color: #f1f5f9;
                padding: 1rem;
                border-radius: 0.5rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            .hidden {
                display: none;
            }

            @media (max-width: 768px) {
                .main-container.fullscreen {
                    grid-template-columns: 1fr;
                    margin: 6rem 1rem 2rem;
                }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="header-content">
                <div class="logo">AI Support Builder</div>
                <button class="get-help-btn" onclick="toggleFullscreen()">Get Help with AI</button>
        </div>
        </header>

        <div class="main-container" id="mainContainer">
            <div class="container" id="builderContainer">
                <h2>Custom Agent Builder</h2>
                <p>Create your own custom support agent tailored to your business needs!</p>
            <button onclick="startAgentBuilder()" class="success-btn">Start Building Custom Agent</button>
            <div id="builderLoading" class="loading">Processing...</div>
            <div id="builderResponse"></div>
        </div>

            <div class="container hidden" id="supportContainer">
                <h2>Customer Support Agent</h2>
                <textarea id="query" placeholder="Type your question here..."></textarea>
                <button onclick="submitQuery()">Submit Query</button>
                <div id="loading" class="loading">Processing...</div>
                <div id="response"></div>
            </div>

            <div class="container" id="testContainer">
            <h2>Test Custom Agent</h2>
            <input type="text" id="agentId" placeholder="Enter Agent ID">
            <textarea id="customQuery" placeholder="Type your question for the custom agent..."></textarea>
            <button onclick="testCustomAgent()">Test Custom Agent</button>
            <div id="customResponse"></div>
            </div>
        </div>

        <script>
            let currentBuilderSession = null;
            let selectedFeatures = [];
            let availableSuggestions = [];
            let featureDetails = {};
            let isFullscreen = false;

            function toggleFullscreen() {
                isFullscreen = !isFullscreen;
                const mainContainer = document.getElementById('mainContainer');
                const supportContainer = document.getElementById('supportContainer');
                
                if (isFullscreen) {
                    mainContainer.classList.add('fullscreen');
                    supportContainer.classList.remove('hidden');
                } else {
                    mainContainer.classList.remove('fullscreen');
                    supportContainer.classList.add('hidden');
                }
            }

            async function submitQuery() {
                const query = document.getElementById('query').value.trim();
                const loading = document.getElementById('loading');
                const response = document.getElementById('response');

                if (!query) {
                    alert('Please enter a query');
                    return;
                }

                loading.style.display = 'block';
                response.innerHTML = '';

                try {
                    const res = await fetch('/support', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query }),
                    });

                    const data = await res.json();
                    if (res.ok) {
                        response.innerHTML = `
                            <div class="response-box">
                                ${data.response}
                            </div>
                        `;
                    } else {
                        response.innerHTML = `<div class="result-box" style="color: red;">Error: ${data.detail}</div>`;
                    }
                } catch (error) {
                    response.innerHTML = `<div class="result-box" style="color: red;">Error: Could not connect to server</div>`;
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function startAgentBuilder() {
                const loading = document.getElementById('builderLoading');
                const response = document.getElementById('builderResponse');
                
                loading.style.display = 'block';
                response.innerHTML = '';
                selectedFeatures = [];
                featureDetails = {};

                try {
                    const res = await fetch('/agent-builder/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({}),
                    });

                    const data = await res.json();
                    if (res.ok) {
                        currentBuilderSession = data.session_id;
                        displayBuilderStep(data);
                    } else {
                        response.innerHTML = `<div class="result-box" style="color: red;">Error: ${data.detail}</div>`;
                    }
                } catch (error) {
                    response.innerHTML = `<div class="result-box" style="color: red;">Error: Could not connect to server</div>`;
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function submitBuilderAnswer(answer) {
                const loading = document.getElementById('builderLoading');
                const response = document.getElementById('builderResponse');
                
                loading.style.display = 'block';

                try {
                    const res = await fetch('/agent-builder/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            session_id: currentBuilderSession, 
                            answer: answer 
                        }),
                    });

                    const data = await res.json();
                    if (res.ok) {
                        if (data.suggestions) {
                            availableSuggestions = data.suggestions;
                        }
                        displayBuilderStep(data);
                    } else {
                        response.innerHTML = `<div class="result-box" style="color: red;">Error: ${data.detail}</div>`;
                    }
                } catch (error) {
                    response.innerHTML = `<div class="result-box" style="color: red;">Error: Could not connect to server</div>`;
                } finally {
                    loading.style.display = 'none';
                }
            }

            function displayBuilderStep(data) {
                const response = document.getElementById('builderResponse');
                
                if (data.is_complete) {
                    response.innerHTML = `
                        <div class="result-box" style="background-color: #d4edda; color: #155724; border-color: #c3e6cb;">
                            <h3>🎉 Agent Created Successfully!</h3>
                            <p><strong>Agent ID:</strong> ${data.agent_id}</p>
                            <p>Your custom agent is ready to use. Copy the Agent ID above and test it in the "Test Custom Agent" section.</p>
                        </div>
                    `;
                } else if (data.step === 0) {
                    // Business name and description input step
                    response.innerHTML = `
                        <div class="result-box">
                            <h3>Step ${data.step + 1}</h3>
                            <p><strong>${data.question}</strong></p>
                            <div class="business-input-container">
                                <input type="text" id="businessNameInput" placeholder="Enter your business name" style="margin-bottom: 10px;">
                                <textarea id="businessDescriptionInput" class="business-description" 
                                    placeholder="Describe what your business does, your main services/products, and how customers typically interact with you..."></textarea>
                                <button onclick="submitBusinessInfo()">Continue</button>
                            </div>
                        </div>
                    `;
                } else if (data.step === 1) {
                    // Features step with custom UI
                    response.innerHTML = `
                        <div class="result-box">
                            <h3>Step ${data.step + 1}</h3>
                            ${data.business_type_info ? `<div class="business-type-detected">${data.business_type_info}</div>` : ''}
                            <p><strong>${data.question}</strong></p>
                            <div class="features-container">
                                <div class="feature-input-container">
                                    <input type="text" id="customFeatureName" class="feature-input" 
                                        placeholder="Enter custom feature name">
                                    <textarea id="customFeatureDescription" class="feature-input"
                                        placeholder="Describe what this feature does (optional)"></textarea>
                                    <button class="add-feature-btn" onclick="addCustomFeature()">Add Custom Feature</button>
                                </div>
                                <div id="selectedFeatures" class="feature-tags"></div>
                                <div class="suggestions">
                                    <p><strong>Suggested features for your business:</strong></p>
                                    <div class="suggestions-grid" id="suggestedFeatures">
                                        Loading suggestions...
                                    </div>
                                </div>
                                <button onclick="submitFeatures()" style="margin-top: 15px;">Continue with Selected Features</button>
                            </div>
                        </div>
                    `;
                    
                    // Load suggestions
                    loadFeatureSuggestions();
                } else {
                    let optionsHtml = '';
                    if (data.options && data.options.length > 0) {
                        optionsHtml = '<div class="options">';
                        data.options.forEach(option => {
                            optionsHtml += `<button class="option-btn" onclick="submitBuilderAnswer('${option}')">${option.replace(/_/g, ' ')}</button>`;
                        });
                        optionsHtml += '</div>';
                    } else {
                        optionsHtml = `
                            <input type="text" id="builderInput" placeholder="Enter your answer...">
                            <button onclick="submitBuilderAnswer(document.getElementById('builderInput').value)">Submit</button>
                        `;
                    }
                    
                    response.innerHTML = `
                        <div class="result-box">
                            <h3>Step ${data.step + 1}</h3>
                            <p><strong>${data.question}</strong></p>
                            ${optionsHtml}
                        </div>
                    `;
                }
            }

            async function loadFeatureSuggestions() {
                try {
                    const res = await fetch(`/agent-builder/features/suggestions?session_id=${currentBuilderSession}`);
                    const data = await res.json();
                    
                    if (res.ok) {
                        const suggestionsContainer = document.getElementById('suggestedFeatures');
                        const selectedContainer = document.getElementById('selectedFeatures');
                        
                        // Display suggested features
                        suggestionsContainer.innerHTML = data.suggested_features.map(feature => `
                            <button class="suggestion-btn" onclick="addSuggestedFeature('${feature.id}')">
                                <strong>${feature.name}</strong>
                                <span class="feature-description">${feature.description}</span>
                                <span class="feature-priority priority-${feature.priority}">${feature.priority}</span>
                            </button>
                        `).join('');
                        
                        // Display selected and custom features
                        updateFeatureDisplay(data.selected_features.concat(data.custom_features));
                    } else {
                        const suggestionsContainer = document.getElementById('suggestedFeatures');
                        suggestionsContainer.innerHTML = '<p style="color: red;">Failed to load suggestions. Please try refreshing the page.</p>';
                    }
                } catch (error) {
                    console.error('Error loading suggestions:', error);
                    const suggestionsContainer = document.getElementById('suggestedFeatures');
                    suggestionsContainer.innerHTML = '<p style="color: red;">Failed to load suggestions. Please try refreshing the page.</p>';
                }
            }

            async function addCustomFeature() {
                const nameInput = document.getElementById('customFeatureName');
                const descInput = document.getElementById('customFeatureDescription');
                const name = nameInput.value.trim();
                const description = descInput.value.trim();
                
                if (!name) {
                    alert('Please enter a feature name');
                    return;
                }
                
                try {
                    const res = await fetch('/agent-builder/features/add-custom', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: currentBuilderSession,
                            feature_name: name,
                            feature_description: description || 'Custom feature added by user'
                        })
                    });
                    
                    if (res.ok) {
                        const data = await res.json();
                        nameInput.value = '';
                        descInput.value = '';
                        loadFeatureSuggestions();  // Refresh the display
                    } else {
                        const data = await res.json();
                        alert(data.detail || 'Failed to add custom feature');
                    }
                } catch (error) {
                    console.error('Error adding custom feature:', error);
                    alert('Failed to add custom feature');
                }
            }

            async function addSuggestedFeature(featureId) {
                try {
                    const res = await fetch(`/agent-builder/features/add-suggested/${currentBuilderSession}/${featureId}`, {
                        method: 'POST'
                    });
                    
                    if (res.ok) {
                        loadFeatureSuggestions();  // Refresh the display
                    } else {
                        const data = await res.json();
                        alert(data.detail || 'Failed to add feature');
                    }
                } catch (error) {
                    console.error('Error adding suggested feature:', error);
                    alert('Failed to add feature');
                }
            }

            async function removeFeature(featureId) {
                try {
                    const res = await fetch(`/agent-builder/features/remove/${currentBuilderSession}/${featureId}`, {
                        method: 'POST'
                    });
                    
                    if (res.ok) {
                        loadFeatureSuggestions();  // Refresh the display
                    } else {
                        const data = await res.json();
                        alert(data.detail || 'Failed to remove feature');
                    }
                } catch (error) {
                    console.error('Error removing feature:', error);
                    alert('Failed to remove feature');
                }
            }

            function updateFeatureDisplay(features) {
                const container = document.getElementById('selectedFeatures');
                container.innerHTML = features.map(feature => `
                    <div class="feature-tag">
                        <div class="feature-info">
                            <strong>${feature.name}</strong>
                            ${feature.description ? `<div class="feature-description">${feature.description}</div>` : ''}
                        </div>
                        <span class="feature-priority priority-${feature.priority}">${feature.priority}</span>
                        <span class="remove-btn" onclick="removeFeature('${feature.id}')">×</span>
                    </div>
                `).join('');
            }

            async function submitFeatures() {
                try {
                    const res = await fetch(`/agent-builder/features/suggestions?session_id=${currentBuilderSession}`);
                    const data = await res.json();
                    
                    if (res.ok) {
                        const allFeatures = data.selected_features.concat(data.custom_features);
                        if (allFeatures.length === 0) {
                            alert('Please add at least one feature');
                            return;
                        }
                        
                        const featureIds = allFeatures.map(f => f.id);
                        submitBuilderAnswer(featureIds.join(','));
                    }
                } catch (error) {
                    console.error('Error submitting features:', error);
                    alert('Failed to submit features');
                }
            }

            function submitBusinessInfo() {
                const name = document.getElementById('businessNameInput').value.trim();
                const description = document.getElementById('businessDescriptionInput').value.trim();
                
                if (!name) {
                    alert('Please enter your business name');
                    return;
                }
                
                submitBuilderAnswer(`${name}\n${description}`);
            }

            async function testCustomAgent() {
                const agentId = document.getElementById('agentId').value.trim();
                const query = document.getElementById('customQuery').value.trim();
                const response = document.getElementById('customResponse');

                if (!agentId || !query) {
                    alert('Please enter both Agent ID and query');
                    return;
                }

                try {
                    const res = await fetch('/custom-agent/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ agent_id: agentId, query: query }),
                    });

                    const data = await res.json();
                    if (res.ok) {
                        response.innerHTML = `
                            <div class="response-box">
                                ${data.response}
                            </div>
                        `;
                    } else {
                        response.innerHTML = `<div class="result-box" style="color: red;">Error: ${data.detail}</div>`;
                    }
                } catch (error) {
                    response.innerHTML = `<div class="result-box" style="color: red;">Error: Could not connect to server</div>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/support", response_model=SupportResponse)
async def get_support(query: Query):
    try:
        state = State(query=query.query)
        result = chain.invoke(state)
        
        # Log analysis results to terminal
        print(f"\nQuery Analysis:")
        print(f"Category: {result['category']}")
        print(f"Sentiment: {result['sentiment']}")
        
        return SupportResponse(
            category=result["category"],
            sentiment=result["sentiment"],
            response=result["response"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/agent-builder/start", response_model=AgentBuilderResponse)
async def start_agent_builder(request: StartAgentBuilderRequest):
    session_id = str(uuid.uuid4())
    agent_builder_sessions[session_id] = AgentBuilderState(session_id=session_id)
    
    return AgentBuilderResponse(
        session_id=session_id,
        step=0,
        question="What is the name of your business? Please also provide a brief description of what your business does.",
        options=None
    )

@app.post("/agent-builder/step", response_model=AgentBuilderResponse)
async def agent_builder_step(request: AgentBuilderStepRequest):
    session = agent_builder_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Process the current step
    if session.current_step == 0:
        # Store business info and analyze context
        parts = request.answer.split("\n", 1)
        session.business_name = parts[0].strip()
        session.business_description = parts[1].strip() if len(parts) > 1 else ""
        
        # Analyze business context
        analysis = analyze_business_context(session.business_name, session.business_description)
        
        session.business_type = analysis["business_type"]
        session.business_context = analysis["business_context"]
        session.key_terms = analysis["key_terms"]
        session.interaction_points = analysis["interaction_points"]
        session.suggested_features = analysis["suggested_features"]  # These are already Feature objects
        
        business_type_info = f"""Based on your description, we understand that:
        
1. Your business type: {analysis['business_type']}
2. Context: {analysis['business_context']}
3. Key interaction points: {', '.join(analysis['interaction_points'])}

We've suggested some features that might be helpful for your customer support needs."""
        
        session.current_step = 1
        return AgentBuilderResponse(
            session_id=session.session_id,
            step=1,
            question="What features should your agent support? Add custom features or select from our suggestions.",
            options=None,
            business_type_info=business_type_info,
            suggestions=[feature.id for feature in session.suggested_features]
        )
    
    elif session.current_step == 1:
        # Handle feature selection/addition
        feature_ids = [f.strip() for f in request.answer.split(",")]
        
        # Combine selected suggested features and custom features
        session.selected_features = [
            feature for feature in session.suggested_features
            if feature.id in feature_ids
        ]
        
        # Add any custom features that weren't in suggestions
        custom_features = [
            Feature(
                id=f_id,
                name=f_id.replace("_", " ").title(),
                description="Custom feature added by user",
                priority=FeaturePriority.MEDIUM,
                is_custom=True
            )
            for f_id in feature_ids
            if f_id not in [f.id for f in session.selected_features]
        ]
        session.selected_features.extend(custom_features)
        
        session.current_step = 2
        return AgentBuilderResponse(
            session_id=session.session_id,
            step=2,
            question="What tone should your agent use?",
            options=["professional", "friendly", "casual", "empathetic"]
        )
    
    elif session.current_step == 2:
        session.tone = request.answer
        session.current_step = 3
        return AgentBuilderResponse(
            session_id=session.session_id,
            step=3,
            question="Any additional requirements or custom instructions for your agent? (Enter text or 'none')",
            options=None
        )
    
    elif session.current_step == 3:
        # Generate the agent
        custom_requirements = request.answer if request.answer.lower() != "none" else ""
        
        # Generate prompts
        prompts = generate_agent_prompts(
            session.business_type,
            session.selected_features,
            session.tone,
            custom_requirements,
            session.business_name
        )
        
        # Create agent configuration
        agent_config = {
            "business_type": session.business_type,
            "features": [feature.dict() for feature in session.selected_features],  # Convert Feature objects to dict
            "tone": session.tone,
            "custom_requirements": custom_requirements,
            "prompts": prompts
        }
        
        # Create and store the agent
        agent_id = str(uuid.uuid4())
        custom_agents[agent_id] = agent_config
        
        session.generated_config = agent_config
        session.is_complete = True
        
        return AgentBuilderResponse(
            session_id=session.session_id,
            step=3,
            question="Agent created successfully!",
            is_complete=True,
            agent_id=agent_id
        )
    
    raise HTTPException(status_code=400, detail="Invalid step")

# API endpoints for feature management
@app.get("/agent-builder/features/suggestions")  # Changed to GET
async def get_session_feature_suggestions(session_id: str):
    """Get current suggestions for the agent builder session"""
    if session_id not in agent_builder_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = agent_builder_sessions[session_id]
    
    # Convert Feature objects to dictionaries
    return {
        "session_id": session_id,
        "current_step": state.current_step,
        "business_type": state.business_type,
        "business_context": state.business_context,
        "suggested_features": [feature.dict() for feature in state.suggested_features],
        "selected_features": [feature.dict() for feature in state.selected_features],
        "custom_features": [feature.dict() for feature in state.custom_features],
        "is_complete": state.is_complete,
        "error": state.error
    }

@app.post("/agent-builder/features/add-suggested/{session_id}/{feature_id}")
async def add_suggested_feature(session_id: str, feature_id: str):
    """Add a suggested feature to the selected features list"""
    if session_id not in agent_builder_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = agent_builder_sessions[session_id]
    
    # Find the suggested feature
    suggested_feature = None
    for feature in state.suggested_features:
        if feature.id == feature_id:
            suggested_feature = feature
            break
    
    if not suggested_feature:
        raise HTTPException(status_code=404, detail="Feature not found in suggestions")
    
    # Check if already selected
    if any(f.id == feature_id for f in state.selected_features):
        raise HTTPException(status_code=400, detail="Feature already selected")
    
    # Add to selected features
    state.selected_features.append(suggested_feature)
    
    return {
        "success": True,
        "selected_features": [feature.dict() for feature in state.selected_features],
        "message": f"Added {suggested_feature.name} to your agent"
    }

@app.post("/agent-builder/features/add-custom")
async def add_custom_feature(request: CustomFeatureRequest):
    """Add a custom feature to the agent"""
    if request.session_id not in agent_builder_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = agent_builder_sessions[request.session_id]
    
    # Create custom feature
    custom_feature = Feature(
        id=f"custom_{len(state.custom_features) + 1}",
        name=request.feature_name,
        description=request.feature_description,
            priority=FeaturePriority.MEDIUM,
        is_custom=True
    )
    
    # Check if feature name already exists
    all_features = state.selected_features + state.custom_features
    if any(f.name.lower() == request.feature_name.lower() for f in all_features):
        raise HTTPException(status_code=400, detail="Feature with this name already exists")
    
    state.custom_features.append(custom_feature)
    
    return {
        "success": True,
        "custom_features": [feature.dict() for feature in state.custom_features],
        "message": f"Added custom feature: {request.feature_name}"
    }

@app.post("/agent-builder/features/remove/{session_id}/{feature_id}")
async def remove_feature(session_id: str, feature_id: str):
    """Remove a feature from selected or custom features"""
    if session_id not in agent_builder_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = agent_builder_sessions[session_id]
    
    # Remove from selected features
    state.selected_features = [f for f in state.selected_features if f.id != feature_id]
    
    # Remove from custom features
    state.custom_features = [f for f in state.custom_features if f.id != feature_id]
    
    return {
        "success": True,
        "selected_features": [feature.dict() for feature in state.selected_features],
        "custom_features": [feature.dict() for feature in state.custom_features],
        "message": "Feature removed successfully"
    }

@app.post("/custom-agent/query", response_model=SupportResponse)
async def query_custom_agent(request: CustomAgentQuery):
    agent_config = custom_agents.get(request.agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail="Custom agent not found")
    
    try:
        # Create custom agent workflow
        custom_workflow = create_custom_agent(agent_config)
        
        # Process query
        state = State(query=request.query)
        result = custom_workflow.invoke(state)
        
        # Log analysis results to terminal
        print(f"\nCustom Agent Analysis:")
        print(f"Agent ID: {request.agent_id}")
        print(f"Category: {result['category']}")
        print(f"Sentiment: {result['sentiment']}")
        
        return SupportResponse(
            category=result["category"],
            sentiment=result["sentiment"],
            response=result["response"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/custom-agents")
async def list_custom_agents():
    """List all created custom agents"""
    return {
        "agents": [
            {
                "id": agent_id,
                "business_type": config["business_type"],
                "features": config["features"],
                "tone": config["tone"]
            }
            for agent_id, config in custom_agents.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)