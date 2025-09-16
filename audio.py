import os
from io import BytesIO
from typing import List
import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs.client import ElevenLabs
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from groq import Groq
import simpleaudio as sa
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from elevenlabs import play
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json


load_dotenv()
elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Initialize Groq client for transcription
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChatGroq for LangChain
llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

class UserDetails(BaseModel):
    """User's personal details extracted from their response."""
    full_name: str = Field(description="The user's full name")
    email: str = Field(description="The user's email address")
    phone_number: str = Field(description="The user's phone number")

def generate_audio(text: str, model: str = "eleven_monolingual_v1", play_locally: bool = False) -> bytes:
    """
    Generate audio from text using ElevenLabs API with English-only model.
    Returns audio bytes. Set play_locally=True to also play locally.
    """
    try:
        import requests

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key or api_key == "sk_your_actual_elevenlabs_key_here":
            print("Error: ELEVENLABS_API_KEY not set or is placeholder")
            return b''

        # Use direct API call with requests for reliability
        # Try Rachel's voice first, fallback to a default voice if not available
        voice_id = "EXAVITQu4vr4xnSDxMaL"  # Rachel's voice
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        print(f"Using ElevenLabs voice ID: {voice_id}")

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }

        data = {
            "text": text,
            "model_id": model,  # eleven_monolingual_v1 for English-only
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        print(f"Making ElevenLabs API request to: {url}")
        print(f"Request data: {data}")

        response = requests.post(url, json=data, headers=headers)
        print(f"ElevenLabs API response status: {response.status_code}")

        if response.status_code != 200:
            print(f"ElevenLabs API error response: {response.text}")

            # If the model/voice combination fails, try with a different model
            if response.status_code == 400 and ("model" in response.text.lower() or "voice" in response.text.lower()):
                print("Trying with eleven_multilingual_v2 model...")
                data["model_id"] = "eleven_multilingual_v2"
                response = requests.post(url, json=data, headers=headers)
                print(f"Retry response status: {response.status_code}")

                if response.status_code != 200:
                    print(f"Retry also failed: {response.text}")

            response.raise_for_status()

        audio_bytes = response.content
        print(f"Received audio data, size: {len(audio_bytes)} bytes")

        if not audio_bytes:
            print("Error: No audio data received from ElevenLabs")
            return b''

        if play_locally and audio_bytes:
            play(audio_bytes)

        return audio_bytes
    except requests.exceptions.RequestException as e:
        print(f"Error with ElevenLabs API request: {e}")
        return b''
    except Exception as e:
        print(f"Error generating audio: {e}")
        return b''

def record_user_input(duration: int = 10, fs: int = 44100) -> BytesIO:
    """
    Record audio from the user's microphone and return it as a BytesIO buffer (WAV format).
    """
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    buffer = BytesIO()
    write(buffer, fs, recording)
    buffer.seek(0)
    return buffer

def extract_text(audio_buffer: BytesIO) -> str:
    """
    Transcribe an audio file to text using Groq's Whisper model.
    """
    try:
        audio_buffer.seek(0)
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_buffer.read()),
            model="whisper-large-v3",
        )
        return transcription.text
    except Exception as e:
        raise Exception(f"Error during transcription: {e}")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    agent_queries: list[str]
    user_details: list[dict[str, str]]
    tech_stack_details: Optional[list[dict[str, str]]]

def speak_and_listen(agent_prompt: str, record_duration=10, backend_mode: bool = False) -> str:
    """
    Generate agent audio, play it, record user response, and transcribe.
    In backend_mode=True, returns audio bytes instead of playing.
    """
    # Generate agent audio
    print(f"Agent Prompt: {agent_prompt}")
    audio_bytes = generate_audio(agent_prompt, play_locally=not backend_mode)

    if backend_mode:
        # In backend mode, don't record - just return the audio bytes as string for now
        # The actual recording will happen on the frontend
        return audio_bytes.decode('latin-1') if isinstance(audio_bytes, bytes) else str(audio_bytes)

    # Record user response
    user_audio = record_user_input(record_duration)
    # Transcribe user response
    user_text = extract_text(user_audio)
    print(f"User: {user_text}")
    return user_text

def greet_node(state: State) -> State:
    """
    Greet the user and return a greeting message using a dynamic agent response.
    """
    prompt="You are a HR Agent, Inform that you are HR Agent. Greet the user and ask if they are looking for a job change in the 'Data Engineering' Role. Be friendly, crisp and short, do not give a lengthy greeting. Never say 'How can I assist you ?'. You are a HR Agent and asking just if they are looking for a job change. "
    res = llm.invoke([HumanMessage(content=prompt)])
    # Add agent's message to state
    state["messages"].append(AIMessage(content=res.content))
    # Speak and listen using the agent's response
    user_response = speak_and_listen(str(res.content), record_duration=3)  # Adjust duration as needed
    # Add user response to state
    state["messages"].append(HumanMessage(content=user_response))
    print("state[messages] in greet node --> ",state["messages"])
    return state

def gather_personal_details_node(state: State) -> State:
    """
    Gather personal details from the user using a dynamic agent response.
    """
    prompt="You are a HR Agent. Ask the user for their Good Name, Email Address and Phone Number. Be polite. Ask for all details at once."
    res = llm.invoke([HumanMessage(content=prompt)])
    # Add agent's message to state
    state["messages"].append(AIMessage(content=res.content))
    # Speak and listen using the agent's response
    user_response = speak_and_listen(str(res.content))
    # Add user response to state
    state["messages"].append(HumanMessage(content=user_response))
    print("state[messages] in gather personal details node --> ",state["messages"])
    user_info_prompt = "Extract the user's Full Name, Email Address and Phone Number from the following text "
    user_info_prompt += f"User Response: {user_response}"
    # Use LLM to extract user details
    user_info = llm.with_structured_output(UserDetails).invoke([HumanMessage(content=user_info_prompt)])
    print("Extracted User Details: ", user_info)
    # Add user details to state
    state["user_details"] = [user_info.model_dump()]
    return state

def get_tech_stack():
    with open('tech_required.txt', 'r') as file:
        base = file.read().strip().split('\n')
        tech_required = base[0].split(':')[1]
        cloud_required = base[1].split(':')[1]
        orchestration_required = base[2].split(':')[1]
    print(f"Tech Required: {tech_required}", f"Cloud Required: {cloud_required}", f"Orchestration Required: {orchestration_required}", sep='\n')
    return tech_required, cloud_required, orchestration_required

class TechStackDetails(BaseModel):
    """
    User's technology stack details.
    """
    tech_stack: str = Field(description="The name of the technology (e.g., Spark)")
    rating: str = Field(description="User's self-assessed rating for this technology")
    explanation: str = Field(description="User's explanation of their experience with this technology")

class TechStackDetailsList(BaseModel):
    items: List[TechStackDetails] = Field(
        description="List of technology stack details provided by the user."
    )

def fetch_user_tech_stack_node(state: State) -> State:
    """
    Fetch the user's technology stack under Data Engineering.
    """
    tech_required, cloud_required, orchestration_required = get_tech_stack()

    # 1. Ask LLM to generate 3 questions, one for each section
    prompt = (
        "You are a HR Agent. For each of the following technology sections, generate a question asking the user if they have experience with the listed technologies, "
        "how they would rate themselves, and any explanation they'd like to add. "
        "Sections:\n"
        f"1. Data Engineering Basics: {tech_required}\n"
        f"2. Cloud (AWS): {cloud_required}\n"
        f"3. Orchestration: {orchestration_required}\n"
        "Return the 3 questions as a list of strings, each question on a new line. "
    )
    llm_questions_res = llm.invoke([HumanMessage(content=prompt)])
    # Assume LLM returns a list of 3 questions as text, parse it
    import ast
    try:
        questions = ast.literal_eval(str(llm_questions_res.content))
    except Exception:
        # fallback: split by newlines if not a list
        questions = [q.strip() for q in str(llm_questions_res.content).split('\n') if q.strip()]

    # 2. Iterate over questions, ask user, collect responses
    user_responses = []
    for question in questions:
        response = speak_and_listen(question, record_duration=20)  # Adjust duration as needed
        user_responses.append(response)
    extraction_prompt = (
        "For each technology mentioned in the following user responses, extract the technology name, the user's self-assessed rating, and any explanation. "
        "Return a dictionary with a single key 'items', whose value is a list of dictionaries, each with keys: 'tech_stack', 'rating', 'explanation'. "
        "If the user mentions multiple technologies in one response, create a separate dictionary for each technology. "
        "Responses:\n"
    )
    for idx, (question, response) in enumerate(zip(questions, user_responses), 1):
        extraction_prompt += f"Section {idx} Question: {question}\nUser Response: {response}\n"
        
    tech_stack_details = llm.with_structured_output(TechStackDetailsList).invoke([HumanMessage(content=extraction_prompt)])
    print("Extracted Tech Stack Details: ", tech_stack_details)
    # Add to state
    with open("user_tech_stack.json", "w") as f:
        json.dump(tech_stack_details.model_dump(), f, indent=2)
    state["tech_stack_details"] = [item.model_dump() for item in tech_stack_details.items]
    return state

def greet_bye(state: State) -> State:
    """
    Greet the user and say goodbye.
    """
    prompt = "You are a HR Agent. Thank the user for their time and say goodbye. Also, inform them that you will reach out to them soon with the next steps. Keep it short and polite."
    res = llm.invoke([HumanMessage(content=prompt)])
    # Add agent's message to state
    state["messages"].append(AIMessage(content=res.content))
    # Speak and listen using the agent's response
    user_response = speak_and_listen(str(res.content))
    # Add user response to state
    state["messages"].append(HumanMessage(content=user_response))
    print("state[messages] in greet bye node --> ",state["messages"])
    return state

def build_hr_agent_graph():
    graph = StateGraph(State)
    # Add nodes
    graph.add_node("greet", greet_node)
    graph.add_node("gather_details", gather_personal_details_node)
    graph.add_node("fetch_tech_stack", fetch_user_tech_stack_node)
    graph.add_node("greet_bye", greet_bye)
    # Define edges (flow)
    graph.add_edge("greet", "gather_details")
    graph.add_edge("gather_details", "fetch_tech_stack")
    graph.add_edge("fetch_tech_stack", "greet_bye")
    # You can add more nodes and edges as needed

    # Set entry point
    graph.set_entry_point("greet")
    return graph

# To run the graph:
if __name__ == "__main__":
    graph = build_hr_agent_graph()
    chain = graph.compile()
    chain.invoke(State(messages=[], agent_queries=[], user_details=[], tech_stack_details=None))