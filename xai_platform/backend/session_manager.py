import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure paths are relative to the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(ROOT_DIR, 'storage')
SESSIONS_FILE = os.path.join(STORAGE_DIR, 'sessions.json')

def _ensure_storage_exists():
    """Ensures that the storage directory and the sessions.json file exist."""
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        logging.info(f"Created storage directory at {STORAGE_DIR}")
        
    if not os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, 'w') as f:
            json.dump([], f)
        logging.info(f"Created new sessions file at {SESSIONS_FILE}")

def load_sessions():
    """
    Loads all previously saved sessions.
    
    Returns:
        list: A list of dictionaries, each representing a session.
    """
    _ensure_storage_exists()
    
    try:
        with open(SESSIONS_FILE, 'r') as f:
            sessions = json.load(f)
            
            # If the file contains a dictionary (e.g., initial placeholder), treat it as empty
            if not isinstance(sessions, list):
                logging.warning(f"{SESSIONS_FILE} contains a {type(sessions).__name__}, not a list. Resetting to empty list.")
                return []
                
            logging.info(f"Loaded {len(sessions)} sessions from {SESSIONS_FILE}")
            return sessions
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {SESSIONS_FILE}. Returning empty list.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading sessions: {e}")
        return []

def save_session(model_name, dataset_name, metrics):
    """
    Saves a new analysis session to the history.
    
    Args:
        model_name (str): The name or identifier of the evaluated model.
        dataset_name (str): The name or identifier of the dataset used.
        metrics (dict): The evaluation metrics computed for the model.
    """
    _ensure_storage_exists()
    
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'dataset_name': dataset_name,
        'metrics': metrics
    }
    
    sessions = load_sessions()
    sessions.append(session_data)
    
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f, indent=4)
        logging.info("Session saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save session: {e}")

if __name__ == "__main__":
    # Small test block
    print("--- Testing Session Manager ---")
    
    # Save a dummy session
    dummy_metrics = {'accuracy': 0.95, 'f1_score': 0.94}
    save_session(model_name="RandomForest_v1", dataset_name="Testing_Data_A", metrics=dummy_metrics)
    
    # Load and print all sessions
    loaded_sessions = load_sessions()
    print(f"\nTotal sessions stored: {len(loaded_sessions)}")
    print("Latest Session:")
    if loaded_sessions:
        print(json.dumps(loaded_sessions[-1], indent=2))
