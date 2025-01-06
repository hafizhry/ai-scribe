import streamlit as st
from openai import OpenAI
import openai
from typing import Tuple
import os

def check_openai_api_key(api_key: str) -> bool:
    """Validate OpenAI API key by attempting to list models."""
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except openai.APIStatusError as e:
        print(e.status_code)
        print(e.response)
        return False
    except Exception as e:
        st.error(f"Error checking API key: {str(e)}")
        return False

def get_api_key(input_key: str) -> Tuple[str, bool]:
    """
    Get and validate the appropriate API key based on input.
    
    Args:
        input_key: User input API key or 'testkey'
        
    Returns:
        Tuple[str, bool]: (actual_key, is_valid)
    """
    if not input_key or input_key.strip() == "testkey":
        # actual_key = st.secrets["OPENAI_API_KEY"] # For dev
        actual_key = os.getenv("OPENAI_API_KEY") # For main
    else:
        actual_key = input_key.strip()
    
    # Validate the key
    is_valid = check_openai_api_key(actual_key)
    return actual_key, is_valid