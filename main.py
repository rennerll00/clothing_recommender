import os
import pandas as pd
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Constants
CHROMA_DB_PATH = "chromadb"
CHROMA_COLLECTION = "autogen-docs-test"
PRODUCT_DATA_PATH = "products.csv"

# Load environment variables securely
load_dotenv()

# Set your OpenAI API key securely
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Helper functions
def is_termination_message(msg):
    """Check if it's a termination message for product search."""
    return "CHECKING PRODUCTS BASED ON" in msg.get("content", "").upper()

def reset_agents(agents):
    """Reset all agents before starting a new conversation."""
    for agent in agents:
        agent.reset()

# Initialize ChromaDB client and collection
def initialize_chroma_db(path, collection_name):
    """Initialize ChromaDB client and collection."""
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_or_create_collection(name=collection_name)

vector_db = initialize_chroma_db(CHROMA_DB_PATH, CHROMA_COLLECTION)

def create_initial_assistant_agent(name, openai_api_key):
    """Create the initial assistant agent responsible for collecting data for recommendations."""
    return ConversableAgent(
        name=name,
        system_message=(
            "You are a helpful clothing product recommendation assistant. Since this is for a specific brand, avoid asking for brand details."
            "Your goal is to understand the user's preferences through their input. If needed, ask clarification questions, but avoid being pushy."
            "Use the information provided to make recommendations, even if the user is not ready to share more details."
            "When its time to retrieve products, pass a list of single, relevant strings representing the user's preferences."
            "If the user mentions specific clothing items (e.g., 'shirt', 'boxers', 'hat'), prioritize those at the beginning of the list."
            "Once you have gathered enough information, respond with 'CHECKING PRODUCTS BASED ON: <user_preferences>', where <user_preferences> is an array of relevant strings."
        ),
        llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
        human_input_mode="ALWAYS",
        is_termination_msg=is_termination_message
    )

def create_rag_proxy_agent(name, vector_db, collection_name):
    """Create the proxy agent for retrieving products."""
    return RetrieveUserProxyAgent(
        name=name,
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "docs_path": [PRODUCT_DATA_PATH],
            "get_or_create": True,
            "overwrite": False,
            "vector_db": vector_db,
            "collection_name": collection_name,
        },
        system_message=(
            "Given the user preferences provided, verify the product catalog, and return products that match."
            "Please consider the fact that the first strings will be clothing pieces such as 'shirt', 'boxers', 'hat' unless that is not specified"
            "Only return products that match the preferences from the catalog; don't make up any products; be inteliggent about how you will do it given these are clothing products"
        ),
        code_execution_config=False
    )

def create_final_assistant_agent(name, openai_api_key):
    """Create the final assistant agent responsible for recommendations."""
    return ConversableAgent(
        name=name,
        system_message=(
            "You are a helpful clothing product recommendation assistant that gets users preferences and also some products data and then recommend up to 3 products from it to our user"
            "Take the list of products retrieved, after the retrieval process is concluded, and undertand the user preferences as well - then make sure you recommend up to 3 products that the user will like the most"
            "Your recommendation must be given in a readable and user-friendly message."
        ),
        llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]},
        human_input_mode="NEVER"
    )
# Group chat setup
def setup_group_chat(agents):
    """Set up a group chat with the specified agents."""
    group_chat = GroupChat(agents=agents, messages=[], max_round=12)
    return GroupChatManager(groupchat=group_chat, llm_config={"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]})

# Main product recommendation flow
def groupchat_product_recommendation_flow(user_input, assistant, ragproxyagent, final_assistant, group_chat_manager):
    """Handle the entire product recommendation flow."""
    try:
        # Step 1: Gather user preferences
        reset_agents([assistant, ragproxyagent, final_assistant])
        assistant_result = assistant.initiate_chat(group_chat_manager, message=user_input)

        # Extract preferences from the assistant's response
        preferences = extract_preferences_from_messages(group_chat_manager.groupchat.messages)
        if not preferences:
            print("Unable to fetch user preferences.")
            return
        print(f"User preferences: {preferences}")

        # Step 2: Retrieve products based on preferences
        retrieval_problem = f"Retrieve products matching these preferences: {preferences}"
        rag_result = ragproxyagent.initiate_chat(group_chat_manager, message=ragproxyagent.message_generator, problem=retrieval_problem)
        
        retrieved_content = get_last_message_content(rag_result)
        if not retrieved_content:
            print("No products retrieved.")
            return
        print(f"Retrieved products: {retrieved_content}")

        # Step 3: Directly finalize recommendations
        formatting_message = f"User preferences: {preferences}\nRetrieved products: {retrieved_content}"
        final_assistant_result = final_assistant.initiate_chat(group_chat_manager, message=formatting_message)

        recommendations = get_last_message_content(final_assistant_result)
        print("Final Product Recommendations:\n", recommendations if recommendations else "No recommendations found.")

    except Exception as e:
        print(f"An error occurred during the recommendation flow: {e}")

# Utility functions
def extract_preferences_from_messages(messages):
    """Extract user preferences from messages."""
    for message in messages:
        if "CHECKING PRODUCTS BASED ON:" in message.get("content", ""):
            return message["content"].split("CHECKING PRODUCTS BASED ON: ")[-1]
    return None

def get_last_message_content(result):
    """Get the content of the last message in the chat history."""
    return result.chat_history[-1].get('content', '') if result and result.chat_history else None

# Main chatbot function
def main():
    print("Welcome to the Product Recommendation Chatbot! Type 'exit' or 'quit' to end the session.")
    initial_assistant = create_initial_assistant_agent("initial_assistant", openai_api_key)
    ragproxyagent = create_rag_proxy_agent("ragproxyagent", vector_db, CHROMA_COLLECTION)
    final_assistant = create_final_assistant_agent("final_assistant", openai_api_key)
    group_chat_manager = setup_group_chat([initial_assistant, ragproxyagent, final_assistant])

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            print("Please enter your preferences.")
            continue

        groupchat_product_recommendation_flow(user_input, initial_assistant, ragproxyagent, final_assistant, group_chat_manager)

# Run chatbot if executed as script
if __name__ == "__main__":
    main()
