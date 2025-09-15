import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

AGENTS_DIR = "agents/configs"
USERS_DIR = "users"
DEFAULT_PROMPTS_FILE = "prompts/default_prompts.json"

def ensure_directory_exists():
    """Ensure the agents and users directories exist"""
    os.makedirs(AGENTS_DIR, exist_ok=True)
    os.makedirs(USERS_DIR, exist_ok=True)

def load_default_prompts(business_type: str = "default") -> Dict[str, str]:
    """Load default prompts for a business type"""
    try:
        with open(DEFAULT_PROMPTS_FILE, 'r') as f:
            prompts = json.load(f)
            return prompts.get(business_type, prompts["default"])
    except FileNotFoundError:
        return {}

def save_agent_config(agent_id: str, config: Dict[str, Any]) -> None:
    """Save agent configuration to JSON file"""
    ensure_directory_exists()
    config["last_modified"] = datetime.now().isoformat()
   
    file_path = os.path.join(AGENTS_DIR, f"{agent_id}.json")
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_agent_config(agent_id: str) -> Optional[Dict[str, Any]]:
    """Load agent configuration from JSON file"""
    file_path = os.path.join(AGENTS_DIR, f"{agent_id}.json")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def update_agent_config(agent_id: str, updates: Dict[str, Any]) -> bool:
    """Update existing agent configuration"""
    config = load_agent_config(agent_id)
    if not config:
        return False
   
    config.update(updates)
    config["last_modified"] = datetime.now().isoformat()
    save_agent_config(agent_id, config)
    return True

def delete_agent_config(agent_id: str) -> bool:
    """Delete agent configuration file"""
    file_path = os.path.join(AGENTS_DIR, f"{agent_id}.json")
    try:
        os.remove(file_path)
        return True
    except FileNotFoundError:
        return False

def list_agents() -> List[str]:
    """List all available agent IDs"""
    ensure_directory_exists()
    return [f.replace('.json', '') for f in os.listdir(AGENTS_DIR) 
            if f.endswith('.json')]

def create_agent_step_1(agent_id: str, business_name: str, business_description: str, 
                       business_type: str = "default") -> Dict[str, Any]:
    """Initialize agent configuration with business info"""
    config = {
        "business_name": business_name,
        "business_description": business_description,
        "business_type": business_type,
        "features": [],
        "custom_features": [],
        "tone": "professional",
        "prompts": load_default_prompts(business_type),
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "current_step": 1,  # Set initial step
        "is_complete": False
    }
    save_agent_config(agent_id, config)
    return config

def add_agent_features(agent_id: str, features: List[Dict[str, Any]]) -> bool:
    """Add features to agent configuration"""
    updates = {
        "features": features,
        "current_step": 2  # Increment step
    }
    return update_agent_config(agent_id, updates)

def set_agent_tone(agent_id: str, tone: str) -> bool:
    """Set agent's communication tone"""
    updates = {
        "tone": tone,
        "current_step": 3  # Increment step
    }
    return update_agent_config(agent_id, updates)

def finalize_agent(agent_id: str, custom_requirements: str = "",
                  contact_email: str = "", contact_phone: str = "") -> bool:
    """Finalize agent configuration with custom settings"""
    updates = {
        "custom_requirements": custom_requirements,
        "contact_info": {
            "email": contact_email,
            "phone": contact_phone
        },
        "is_complete": True,
        "current_step": 4  # Set final step
    }
    return update_agent_config(agent_id, updates)

# User Management Functions
def create_user_session() -> str:
    """Create a new user session ID"""
    return str(uuid.uuid4())

def save_user_prompt(user_id: str, prompt_id: str, prompt_data: Dict[str, Any]) -> None:
    """Save a user's prompt as JSON"""
    ensure_directory_exists()
    user_dir = os.path.join(USERS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    prompt_data["created_at"] = datetime.now().isoformat()
    prompt_data["last_modified"] = datetime.now().isoformat()

    file_path = os.path.join(user_dir, f"{prompt_id}.json")
    with open(file_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)

def load_user_prompt(user_id: str, prompt_id: str) -> Optional[Dict[str, Any]]:
    """Load a user's prompt from JSON"""
    file_path = os.path.join(USERS_DIR, user_id, f"{prompt_id}.json")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def list_user_prompts(user_id: str) -> List[Dict[str, Any]]:
    """List all prompts for a user"""
    user_dir = os.path.join(USERS_DIR, user_id)
    if not os.path.exists(user_dir):
        return []

    prompts = []
    for file in os.listdir(user_dir):
        if file.endswith('.json'):
            prompt_id = file.replace('.json', '')
            prompt_data = load_user_prompt(user_id, prompt_id)
            if prompt_data:
                prompt_data["prompt_id"] = prompt_id
                prompts.append(prompt_data)
    return prompts

def update_user_prompt(user_id: str, prompt_id: str, updates: Dict[str, Any]) -> bool:
    """Update a user's prompt"""
    prompt_data = load_user_prompt(user_id, prompt_id)
    if not prompt_data:
        return False

    prompt_data.update(updates)
    prompt_data["last_modified"] = datetime.now().isoformat()
    save_user_prompt(user_id, prompt_id, prompt_data)
    return True

def delete_user_prompt(user_id: str, prompt_id: str) -> bool:
    """Delete a user's prompt"""
    file_path = os.path.join(USERS_DIR, user_id, f"{prompt_id}.json")
    try:
        os.remove(file_path)
        return True
    except FileNotFoundError:
        return False

def extract_menu_data(description: str) -> Dict[str, Any]:
    """Extract menu items and pricing from business description"""
    menu_data = {
        "menu_items": [],
        "pricing_info": "",
        "special_offers": ""
    }

    # Simple extraction logic - can be enhanced with LLM if needed
    lines = description.split('\n')
    current_section = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for menu items (lines containing prices or food items)
        if '$' in line or 'price' in line.lower() or any(word in line.lower() for word in ['pizza', 'burger', 'pasta', 'salad', 'drink', 'dessert']):
            menu_data["menu_items"].append(line)
        elif 'delivery' in line.lower() or 'fee' in line.lower():
            menu_data["pricing_info"] += line + " "
        elif 'offer' in line.lower() or 'discount' in line.lower() or 'deal' in line.lower():
            menu_data["special_offers"] += line + " "

    return menu_data

def update_agent_with_menu_data(agent_id: str, description: str) -> bool:
    """Update agent configuration with extracted menu data"""
    menu_data = extract_menu_data(description)
    updates = {
        "menu_data": menu_data,
        "business_description": description
    }
    return update_agent_config(agent_id, updates)