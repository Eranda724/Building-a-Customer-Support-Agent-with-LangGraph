from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, SecretStr
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
import uuid
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Import our custom modules
from custom_agent_engine import State, build_agent_workflow
from agent_storage import (
    save_agent_config,
    load_agent_config,
    create_agent_step_1,
    add_agent_features,
    set_agent_tone,
    finalize_agent,
    load_default_prompts
)

# Load environment variables
load_dotenv()

# Get API key and validate
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

# Initialize LLM with proper API key
llm = ChatGroq(
    temperature=0.7,
    api_key=GROQ_API_KEY,
    model="mistral-saba-24b")

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

# Keep existing endpoints
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    # Keep the existing UI HTML code here
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
        question="What is the name of your business? Please also provide a brief description of what your business does.",
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
            
            # Create initial agent config
            config = create_agent_step_1(
                request.session_id,
                business_name,
                business_description
            )
            
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
                options=["professional", "friendly", "casual", "empathetic"]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)