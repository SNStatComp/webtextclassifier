from .llm_api import LLM_API


def classify(config, extracted_text):
    # For now, just prompt the llm
    return LLM_API(config, extracted_text)
