from .llm_api import prompt_LLM
from .llm_embedder import embed_sentence_transformer

# TODO use for classification
from sklearn.linear_model import LogisticRegression


def classify_prompt(config, extracted_text):
    # For now, just prompt the llm
    return prompt_LLM(config, extracted_text)


# TODO resturcture code to pass pre-trained embedder/classifier
def classify_embedding(config, extracted_text):
    embedding = embed_sentence_transformer(config, extracted_text)

    # TODO replace this with something that makes sense
    return "Yes" if abs(sum(embedding)) >= 0.5 else "No"


def classify(config, extracted_text):
    match config.llm.method:
        case "prompt":
            return classify_prompt(config, extracted_text)
        case "embedding":
            return classify_embedding(config, extracted_text)
        case _:
            raise Exception(f"Unspecified method for classification: {config.llm.method}")
