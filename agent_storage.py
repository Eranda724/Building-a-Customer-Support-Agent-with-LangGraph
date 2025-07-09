import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import uuid

AGENTS_DIR = Path("agents/configs")

def ensure_agents_dir():
    """Ensure the agents directory exists"""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

def save_agent_config(agent_id: str, config: Dict[str, Any]) -> None:
    """Save agent configuration to a JSON file"""
    ensure_agents_dir()
   
    # Add metadata
    config["metadata"] = {
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "agent_id": agent_id
    }
   
    file_path = AGENTS_DIR / f"{agent_id}.json"
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)

def load_agent_config(agent_id: str) -> Optional[Dict[str, Any]]:
    """Load agent configuration from JSON file"""
    file_path = AGENTS_DIR / f"{agent_id}.json"
    if not file_path.exists():
        return None
   
    with open(file_path, "r") as f:
        return json.load(f)

def update_agent_config(agent_id: str, updates: Dict[str, Any]) -> bool:
    """Update existing agent configuration"""
    config = load_agent_config(agent_id)
    if not config:
        return False
   
    # Update configuration
    config.update(updates)
    config["metadata"]["last_modified"] = datetime.now().isoformat()
   
    # Save updated config
    save_agent_config(agent_id, config)
    return True

def delete_agent_config(agent_id: str) -> bool:
    """Delete agent configuration file"""
    file_path = AGENTS_DIR / f"{agent_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False

def list_agents() -> List[Dict[str, Any]]:
    """List all available agents"""
    ensure_agents_dir()
    agents = []
   
    for file_path in AGENTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
                agents.append({
                    "id": config["metadata"]["agent_id"],
                    "business_name": config.get("business_name", "Unknown"),
                    "business_type": config.get("business_type", "custom"),
                    "features": [f["name"] for f in config.get("custom_features", [])],
                    "created_at": config["metadata"]["created_at"]
                })
        except Exception as e:
            print(f"Error loading agent config {file_path}: {e}")
            continue
   
    return agents

def create_agent_step_1(business_name: str, business_purpose: str, business_description: str) -> str:
    """Step 1: Create agent with business information"""
    agent_id = str(uuid.uuid4())
    
    config = {
        "business_type": "custom",
        "business_name": business_name,
        "business_purpose": business_purpose,
        "business_description": business_description,
        "custom_features": [],
        "tone": "professional",
        "custom_requirements": "",
        "contact_info": {
            "email": "",
            "phone": ""
        },
        "current_step": 1,
        "is_complete": False
    }
    
    save_agent_config(agent_id, config)
    return agent_id

def add_agent_features(agent_id: str, features: List[Dict[str, str]]) -> bool:
    """Step 2: Add features to agent"""
    config = load_agent_config(agent_id)
    if not config:
        return False
    
    # Convert user features to proper format
    custom_features = []
    for i, feature in enumerate(features):
        custom_features.append({
            "id": f"custom_{i}",
            "name": feature["name"],
            "description": feature["description"],
            "priority": "medium",
            "is_custom": True
        })
    
    config["custom_features"] = custom_features
    config["current_step"] = 2
    
    save_agent_config(agent_id, config)
    return True

def set_agent_tone(agent_id: str, tone: str) -> bool:
    """Step 3: Set agent tone"""
    config = load_agent_config(agent_id)
    if not config:
        return False
    
    config["tone"] = tone
    config["current_step"] = 3
    
    save_agent_config(agent_id, config)
    return True

def finalize_agent(agent_id: str, custom_requirements: str = "", contact_email: str = "", contact_phone: str = "") -> bool:
    """Step 4: Finalize agent with custom requirements and contact info"""
    config = load_agent_config(agent_id)
    if not config:
        return False
    
    config["custom_requirements"] = custom_requirements
    config["contact_info"]["email"] = contact_email
    config["contact_info"]["phone"] = contact_phone
    config["current_step"] = 4
    config["is_complete"] = True
    
    # Generate prompts based on user inputs
    config["prompts"] = generate_prompts(config)
    
    # Generate business context and key terms
    config["business_context"] = f"Business providing {config['business_name']} - {config['business_purpose']} services"
    config["key_terms"] = extract_key_terms(config)
    
    save_agent_config(agent_id, config)
    return True

def generate_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """Generate all prompt templates based on user configuration"""
    
    # Build features list for prompts
    features_text = ""
    if config["custom_features"]:
        features_text = "\n".join([f"- {f['name']}: {f['description']}" for f in config["custom_features"]])
    
    # Build contact info text
    contact_text = f"Email: {config['contact_info']['email']}\nPhone: {config['contact_info']['phone']}"
    
    # Base prompt template
    base_prompt = f"""You are a professional customer support agent for {config['business_name']} - {config['business_purpose']}.

Always maintain a {config['tone']} tone in your responses.

IMPORTANT: Use only plain text formatting. Do not use markdown symbols like #, *, or **.
Use simple text with proper spacing, clear sections, and easy-to-read formatting.

Available Features:
{features_text}

Contact Information:
{contact_text}

Additional guidelines:
- Always greet the customer politely
- Be concise but thorough in your responses
- If you don't know an answer, offer to connect the customer with a specialist
- {config['custom_requirements']}

Your goal is to provide excellent customer support and resolve issues efficiently for {config['business_name']}."""

    return {
        "categorize": f"""{base_prompt}

Categorize this customer query into the most appropriate category for {config['business_name']}.
Available categories: Technical, Billing, General, or Other if it doesn't fit.
Respond with ONLY the category name.
Query: {{query}}""",
        
        "technical": f"""{base_prompt}

Provide detailed technical support for this {config['business_name']} query.
Be sure to:
- Acknowledge the customer's issue
- Provide step-by-step guidance if applicable
- Offer additional help if the issue isn't resolved
Use plain text formatting only - no markdown symbols.
Query: {{query}}""",
        
        "billing": f"""{base_prompt}

Handle this billing/payment related query for {config['business_name']}.
Important notes:
- Never share sensitive customer information
- Be clear about payment policies
- Provide relevant pricing information from available features
- Offer to connect with accounting for complex issues
Use plain text formatting only - no markdown symbols.
Query: {{query}}""",
        
        "general": f"""{base_prompt}

Provide comprehensive general customer support for this {config['business_name']} query.
Remember to:
- Be friendly and welcoming
- Provide complete information about available features
- If the query doesn't match any specific feature, provide contact information
- Offer additional assistance
Use plain text formatting only - no markdown symbols.
Query: {{query}}""",
        
        "escalate": f"""{base_prompt}

This query needs to be escalated to a human agent.
Politely inform the customer that their issue will be handled by a specialist.
Provide contact information: {config['contact_info']['email']} and {config['contact_info']['phone']}
Provide an estimated wait time if possible.
Use plain text formatting only - no markdown symbols.
Query: {{query}}"""
    }

def extract_key_terms(config: Dict[str, Any]) -> List[str]:
    """Extract key terms from business information"""
    terms = []
    
    # Add business name words
    terms.extend(config["business_name"].lower().split())
    
    # Add purpose words
    terms.extend(config["business_purpose"].lower().split())
    
    # Add feature names
    for feature in config["custom_features"]:
        terms.extend(feature["name"].lower().split())
    
    # Remove common words and duplicates
    common_words = {"the", "a", "an", "and", "or", "but", "for", "with", "to", "of", "in", "on", "at", "by"}
    terms = list(set([term for term in terms if term not in common_words and len(term) > 2]))
    
    return terms

def get_suggested_tone(business_purpose: str) -> List[str]:
    """Suggest appropriate tones based on business purpose"""
    purpose_lower = business_purpose.lower()
    
    if any(word in purpose_lower for word in ["fitness", "gym", "training", "sport"]):
        return ["casual", "friendly"]
    elif any(word in purpose_lower for word in ["medical", "health", "doctor", "clinic"]):
        return ["empathetic", "professional"]
    elif any(word in purpose_lower for word in ["restaurant", "food", "cafe", "dining"]):
        return ["friendly", "casual"]
    elif any(word in purpose_lower for word in ["legal", "law", "attorney", "finance"]):
        return ["professional", "formal"]
    else:
        return ["friendly", "professional"]

# Example usage functions
def create_gymbuddy_example():
    """Create the GymBuddy Pro example"""
    
    # Step 1: Basic info
    agent_id = create_agent_step_1(
        business_name="GymBuddy Pro",
        business_purpose="customer calling agent for get schedule from gym trainer",
        business_description="GymBuddy Pro is a comprehensive fitness application designed to help you achieve your fitness goals with personalized workouts and expert guidance. Whether you're a beginner or an advanced fitness enthusiast, GymBuddy Pro provides a tailored experience to suit your needs."
    )
    
    # Step 2: Add features
    features = [
        {
            "name": "Pricing",
            "description": "Multiple packages available: 1 day = Rs. 300, 1 week = Rs. 1000, 1 month = Rs. 5000"
        },
        {
            "name": "Location",
            "description": "I can come to home, office, online sessions available. For equipment and weights training, let's meet at the gym."
        }
    ]
    add_agent_features(agent_id, features)
    
    # Step 3: Set tone
    set_agent_tone(agent_id, "casual")
    
    # Step 4: Finalize
    finalize_agent(
        agent_id=agent_id,
        custom_requirements="When users ask about discounts, provide email for special offers. For scheduling and technical issues, provide phone number for direct contact.",
        contact_email="support@gymbuddypro.com",
        contact_phone="+1-800-GYM-BUDDY"
    )
    
    return agent_id