"""This module contains the detect_language function, which makes a call to the Google Cloud Translation API to detect the language of a text."""

# Import the Google Cloud client library
from google.cloud import translate_v2 as translate

# Other imports
import json
from typing import Dict, Text

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./credentials.json"

def detect_language(text: str) -> Text:
    """Detects the text's language."""

    detect_client = translate.Client()

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = detect_client.detect_language(text)

    return result["language"]

