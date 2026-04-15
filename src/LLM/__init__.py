from .llm_classifier import create_classification_model, classify_prompt, classify_embedding
from .llm_embedder import create_embedding_model
import warnings


# Generic function to create models
def create_model(config, model_type):
    match model_type:
        case "embedder":
            if (config.llm.embedding.model_weights is not None):
                return create_embedding_model(config, local=True)
            else:
                return create_embedding_model(config)
        case "classifier":
            return create_classification_model(config)
        case _:
            warnings.warn(f"Model type {model_type} has no defined model creation method!")
            return None


# Generic function to classify text
def classify(config, extracted_text, variable=None, models=None):
    match config.llm.method:
        case "prompt":
            return classify_prompt(config, extracted_text, variable)
        case "embedding":
            return classify_embedding(config, extracted_text, models)
        case _:
            raise Exception(f"Unspecified method for classification: {config.llm.method}")