from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import uuid
import os
from dotenv import load_dotenv
from io import BytesIO
import base64
from audio import extract_text, generate_audio
import ssl
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import ipaddress

# Import our custom modules
from custom_agent_engine import State, build_agent_workflow
from agent_storage import (
    save_agent_config,
    load_agent_config,
    create_agent_step_1,
    add_agent_features,
    set_agent_tone,
    finalize_agent,
    load_default_prompts,
    update_agent_with_menu_data,
    create_user_session,
    save_user_prompt,
    load_user_prompt,
    list_user_prompts,
    update_user_prompt,
    delete_user_prompt
)

# Load environment variables
load_dotenv()

# Generate self-signed certificates for HTTPS if not exist
if not os.path.exists('key.pem') or not os.path.exists('cert.pem'):
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Write private key
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Write certificate
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

# Get API keys and validate
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Please create a .env file with your Groq API key (GROQ_API_KEY=gsk_...)"
    )

if not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError(
        "Invalid GROQ_API_KEY format. "
        "The key should start with 'gsk_'. "
        "Please check your API key at https://console.groq.com/"
    )

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "sk_your_actual_elevenlabs_key_here":
    raise ValueError(
        "ELEVENLABS_API_KEY not found or is placeholder in environment variables. "
        "Please create a .env file with your actual ElevenLabs API key from https://elevenlabs.io/app/profile"
    )

# Initialize LLM with proper API key
llm = ChatGroq(
    temperature=0.7,
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Support Agent API",
    description="AI-powered customer support system",
    version="1.0.0"
)

# Add error handling middleware
@app.middleware("http")
async def handle_exceptions(request, call_next):
    try:
        return await call_next(request)
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            return JSONResponse(
                status_code=500,
                content={
                    "detail": str(e),
                    "help": "Please check your .env file and ensure you have added your Groq API key correctly."
                }
            )
        raise
    except Exception as e:
        if "invalid_api_key" in str(e).lower():
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid Groq API key. Please check your .env file and ensure your API key is correct.",
                    "help": "Get your API key from https://console.groq.com/"
                }
            )
        raise

# Configure CORS
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

class AgentBuilderStepRequest(BaseModel):
    session_id: str
    answer: str

class CustomAgentQuery(BaseModel):
    agent_id: str
    query: str

class CustomFeatureRequest(BaseModel):
    session_id: str
    feature_name: str
    feature_description: str = "Custom feature added by user"

class FeatureSuggestionsResponse(BaseModel):
    suggested_features: List[Dict[str, Any]]
    selected_features: List[Dict[str, Any]]
    custom_features: List[Dict[str, Any]]

# Default feature suggestions by business type
DEFAULT_FEATURES = {
    "fitness_and_wellness": [
        {
            "id": "schedule_booking",
            "name": "Schedule Booking",
            "description": "Book and manage training sessions",
            "priority": "high"
        },
        {
            "id": "trainer_info",
            "name": "Trainer Information",
            "description": "Information about trainers and specialties",
            "priority": "medium"
        },
        {
            "id": "pricing_info",
            "name": "Pricing Information",
            "description": "Details about training packages and pricing",
            "priority": "high"
        }
    ],
    "ecommerce": [
        {
            "id": "order_tracking",
            "name": "Order Tracking",
            "description": "Track order status and delivery",
            "priority": "high"
        },
        {
            "id": "product_info",
            "name": "Product Information",
            "description": "Details about products and availability",
            "priority": "high"
        },
        {
            "id": "payment_support",
            "name": "Payment Support",
            "description": "Handle payment and refund queries",
            "priority": "medium"
        }
    ],
    "food_delivery": [
        {
            "id": "menu_inquiry",
            "name": "Menu Information",
            "description": "Provide details about menu items and availability",
            "priority": "high"
        },
        {
            "id": "pricing_info",
            "name": "Pricing Information",
            "description": "Handle pricing and delivery fee queries",
            "priority": "high"
        },
        {
            "id": "order_placement",
            "name": "Order Placement",
            "description": "Assist with placing and modifying orders",
            "priority": "high"
        },
        {
            "id": "delivery_tracking",
            "name": "Delivery Tracking",
            "description": "Track order status and delivery information",
            "priority": "medium"
        }
    ],
    "default": [
        {
            "id": "general_support",
            "name": "General Support",
            "description": "Basic customer support capabilities",
            "priority": "high"
        },
        {
            "id": "contact_management",
            "name": "Contact Management",
            "description": "Handle customer inquiries and communication",
            "priority": "medium"
        }
    ]
}

# Feature management endpoints
@app.get("/agent-builder/features/suggestions", response_model=FeatureSuggestionsResponse)
async def get_feature_suggestions(session_id: str):
    """Get suggested and selected features for an agent"""
    config = load_agent_config(session_id)
    if not config:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get default features based on business type
    business_type = config.get("business_type", "default")
    suggested_features = DEFAULT_FEATURES.get(business_type, DEFAULT_FEATURES["default"])
    
    return FeatureSuggestionsResponse(
        suggested_features=suggested_features,
        selected_features=config.get("features", []),
        custom_features=config.get("custom_features", [])
    )

@app.post("/agent-builder/features/add-custom")
async def add_custom_feature(request: CustomFeatureRequest):
    """Add a custom feature to an agent"""
    config = load_agent_config(request.session_id)
    if not config:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create new custom feature
    custom_features = config.get("custom_features", [])
    new_feature = {
        "id": f"custom_{len(custom_features)}",
        "name": request.feature_name,
        "description": request.feature_description,
        "priority": "medium",
        "is_custom": True
    }
    
    # Add to custom features
    custom_features.append(new_feature)
    config["custom_features"] = custom_features
    
    # Also add to selected features
    selected_features = config.get("features", [])
    selected_features.append(new_feature)
    config["features"] = selected_features
    
    # Save updated config
    save_agent_config(request.session_id, config)
    
    return {"status": "success", "feature": new_feature}

@app.post("/agent-builder/features/add-suggested/{session_id}/{feature_id}")
async def add_suggested_feature(session_id: str, feature_id: str):
    """Add a suggested feature to selected features"""
    config = load_agent_config(session_id)
    if not config:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get default features based on business type
    business_type = config.get("business_type", "default")
    suggested_features = DEFAULT_FEATURES.get(business_type, DEFAULT_FEATURES["default"])
    
    # Find the suggested feature
    suggested_feature = next(
        (f for f in suggested_features if f["id"] == feature_id),
        None
    )
    if not suggested_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    
    # Add to selected features if not already there
    selected_features = config.get("features", [])
    if not any(f["id"] == feature_id for f in selected_features):
        selected_features.append(suggested_feature)
        config["features"] = selected_features
        save_agent_config(session_id, config)
    
    return {"status": "success", "feature": suggested_feature}

@app.post("/agent-builder/features/remove/{session_id}/{feature_id}")
async def remove_feature(session_id: str, feature_id: str):
    """Remove a feature from selected features"""
    config = load_agent_config(session_id)
    if not config:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove from selected features
    selected_features = config.get("features", [])
    config["features"] = [f for f in selected_features if f["id"] != feature_id]
    
    # If it's a custom feature, also remove from custom features
    if feature_id.startswith("custom_"):
        custom_features = config.get("custom_features", [])
        config["custom_features"] = [f for f in custom_features if f["id"] != feature_id]
    
    save_agent_config(session_id, config)
    return {"status": "success"}

@app.get("/")
async def get_ui():
    return FileResponse("index.html", media_type="text/html")

@app.post("/support", response_model=SupportResponse)
async def get_support(query: Query):
    try:
        state = State(query=query.query)
        # The workflow is now built and invoked within the custom_agent_engine module
        # We need to load the default prompts and then build the workflow
        default_prompts = load_default_prompts()
        workflow = build_agent_workflow(default_prompts)
        result = workflow.invoke(state)
        
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
    """Start the agent builder process"""
    session_id = str(uuid.uuid4())

    return AgentBuilderResponse(
        session_id=session_id,
        step=0,
        question="What is the name of your business/system? Please provide a detailed description of what it does, its services, or any specific information customers should know.",
        options=None
    )

@app.post("/agent-builder/step", response_model=AgentBuilderResponse)
async def agent_builder_step(request: AgentBuilderStepRequest):
    """Handle each step of the agent builder process"""
    agent_config = load_agent_config(request.session_id)
    current_step = agent_config.get("current_step", 0) if agent_config else 0
    
    try:
        if current_step == 0:
            # Process business info
            parts = request.answer.split("\n", 1)
            business_name = parts[0].strip()
            business_description = parts[1].strip() if len(parts) > 1 else ""

            # Create initial agent config with universal type
            config = create_agent_step_1(
                request.session_id,
                business_name,
                business_description,
                business_type="universal"  # Use universal type for any business
            )

            # Extract menu data for food delivery businesses
            if "food" in business_description.lower() or "restaurant" in business_description.lower() or "delivery" in business_description.lower():
                update_agent_with_menu_data(request.session_id, business_description)

            # Save user's prompt data
            prompt_data = {
                "business_name": business_name,
                "business_description": business_description,
                "business_type": "universal",
                "session_id": request.session_id
            }
            # Use session_id as user_id for now (can be enhanced with proper user auth later)
            save_user_prompt(request.session_id, request.session_id, prompt_data)
            
            return AgentBuilderResponse(
                session_id=request.session_id,
                step=1,
                question="What features should your agent support? Add custom features or select from our suggestions.",
                options=None,
                business_type_info=f"Business Type: {config['business_type']}"
            )
            
        elif current_step == 1:
            # Process features
            features = []
            for feature_id in request.answer.split(","):
                feature_id = feature_id.strip()
                features.append({
                    "id": feature_id,
                    "name": feature_id.replace("_", " ").title(),
                    "description": f"Support for {feature_id.replace('_', ' ')}",
                    "priority": "medium"
                })
            
            add_agent_features(request.session_id, features)
            
            return AgentBuilderResponse(
                session_id=request.session_id,
                step=2,
                question="What tone should your agent use?",
                options=["professional - Formal and business-like communication", "friendly - Warm and approachable tone", "casual - Relaxed and conversational style", "empathetic - Understanding and caring approach"]
            )
            
        elif current_step == 2:
            # Process tone selection
            set_agent_tone(request.session_id, request.answer)
            
            return AgentBuilderResponse(
                session_id=request.session_id,
                step=3,
                question="Any additional requirements or custom instructions for your agent? (Enter text or 'none')",
                options=None
            )
            
        elif current_step == 3:
            # Finalize agent
            custom_requirements = request.answer if request.answer.lower() != "none" else ""
            
            finalize_agent(
                request.session_id,
                custom_requirements=custom_requirements,
                contact_email="support@example.com",
                contact_phone="1-800-SUPPORT"
            )
            
            return AgentBuilderResponse(
                session_id=request.session_id,
                step=4,
                question="Agent created successfully!",
                is_complete=True,
                agent_id=request.session_id
            )
            
        raise HTTPException(status_code=400, detail="Invalid step")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing step: {str(e)}")

@app.post("/custom-agent/query")
async def query_custom_agent(query: CustomAgentQuery):
    """Handle queries for custom agents"""
    # Load agent configuration
    config = load_agent_config(query.agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        # Create state for query
        state = State(query=query.query)

        # Build and run workflow
        workflow = build_agent_workflow(config)
        result = workflow.invoke(state)

        return {
            "response": result["response"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/custom-agent/voice-query/{agent_id}")
async def voice_query_custom_agent(agent_id: str, audio_file: UploadFile = File(...)):
    """Handle voice queries for custom agents"""
    # Load agent configuration
    config = load_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        # Read audio file
        audio_content = await audio_file.read()
        audio_buffer = BytesIO(audio_content)

        # Transcribe audio using Groq Whisper
        query_text = extract_text(audio_buffer)

        # Create state for query
        state = State(query=query_text)

        # Build and run workflow
        workflow = build_agent_workflow(config)
        result = workflow.invoke(state)

        # Generate audio response using ElevenLabs
        try:
            print(f"Generating audio for response: {result['response'][:100]}...")
            audio_response_bytes = generate_audio(result["response"])

            # Check if audio generation was successful
            if not audio_response_bytes:
                print("Error: generate_audio returned empty bytes")
                raise HTTPException(status_code=500, detail="Failed to generate audio response - no audio data received")

            print(f"Audio generated successfully, size: {len(audio_response_bytes)} bytes")

            # Convert to base64 for JSON response
            audio_base64 = base64.b64encode(audio_response_bytes).decode()
            print(f"Audio converted to base64, length: {len(audio_base64)}")
        except Exception as audio_error:
            print(f"Audio generation error: {str(audio_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate audio response: {str(audio_error)}")

        return {
            "transcribed_query": query_text,
            "response": result["response"],
            "audio_response": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice query: {str(e)}")

@app.post("/custom-agent/generate-voice")
async def generate_voice_response(query: CustomAgentQuery):
    """Generate voice response for custom agents"""
    # Load agent configuration
    config = load_agent_config(query.agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        # Create state for query
        state = State(query=query.query)

        # Build and run workflow
        workflow = build_agent_workflow(config)
        result = workflow.invoke(state)

        # Generate audio response (placeholder)
        audio_data = base64.b64encode(b"audio_placeholder").decode()

        return {
            "response": result["response"],
            "audio_base64": audio_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating voice response: {str(e)}")

# CustomGPT - Guided Prompt Generation
class GuidedPromptRequest(BaseModel):
    query: str

@app.post("/customgpt/generate-guide")
async def generate_business_guide(request: GuidedPromptRequest):
    """Generate guided prompts for creating customer support agents"""
    try:
        # Create a prompt to generate business guide
        guide_prompt = f"""
        You are an expert business consultant. A user wants to create a customer support agent for: "{request.query}"

        Please provide a comprehensive guide with:
        1. Business Name suggestion
        2. Detailed business description (2-3 sentences)
        3. Key features the agent should support (5-8 features)
        4. Suggested agent tone
        5. Any special considerations

        Format your response as a JSON object with these keys:
        - business_name
        - business_description
        - features (array of feature objects with 'name' and 'description')
        - suggested_tone
        - special_considerations

        Make it practical and ready to use for building a customer support agent.
        """

        response = llm.invoke([HumanMessage(content=guide_prompt)])

        # Try to parse as JSON, if not, create structured response
        try:
            import json
            guide_data = json.loads(response.content.strip())
        except:
            # Fallback: extract information from text response
            content = response.content
            guide_data = {
                "business_name": "Generated Business Name",
                "business_description": content[:200] + "...",
                "features": [
                    {"name": "General Support", "description": "Basic customer support"},
                    {"name": "Information", "description": "Provide business information"}
                ],
                "suggested_tone": "professional",
                "special_considerations": "Generated guide for " + request.query
            }

        return guide_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating guide: {str(e)}")

# User Management Endpoints
@app.post("/user/session")
async def create_session():
    """Create a new user session"""
    session_id = create_user_session()
    return {"session_id": session_id}

@app.post("/user/prompt/save")
async def save_prompt(request: dict):
    """Save a user's prompt"""
    user_id = request.get("user_id")
    prompt_id = request.get("prompt_id")
    prompt_data = request.get("prompt_data", {})

    if not user_id or not prompt_id:
        raise HTTPException(status_code=400, detail="user_id and prompt_id are required")

    try:
        save_user_prompt(user_id, prompt_id, prompt_data)
        return {"status": "success", "message": "Prompt saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving prompt: {str(e)}")

@app.get("/user/prompts/{user_id}")
async def get_user_prompts(user_id: str):
    """Get all prompts for a user"""
    try:
        prompts = list_user_prompts(user_id)
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompts: {str(e)}")

@app.get("/user/prompt/{user_id}/{prompt_id}")
async def get_user_prompt(user_id: str, prompt_id: str):
    """Get a specific user prompt"""
    prompt_data = load_user_prompt(user_id, prompt_id)
    if not prompt_data:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return prompt_data

@app.put("/user/prompt/{user_id}/{prompt_id}")
async def update_prompt(user_id: str, prompt_id: str, updates: dict):
    """Update a user's prompt"""
    success = update_user_prompt(user_id, prompt_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return {"status": "success", "message": "Prompt updated successfully"}

@app.delete("/user/prompt/{user_id}/{prompt_id}")
async def delete_prompt(user_id: str, prompt_id: str):
    """Delete a user's prompt"""
    success = delete_user_prompt(user_id, prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return {"status": "success", "message": "Prompt deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')
    uvicorn.run(app, host="0.0.0.0", port=8443, ssl=ssl_context)