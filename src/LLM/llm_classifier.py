from .llm_api import prompt_LLM
from .llm_embedder import embed_text


def classify_prompt(config, extracted_text):
    # For now, just prompt the llm
    return prompt_LLM(config, extracted_text)


# TODO restructure code to process in batches
def classify_embedding(config, extracted_text, models):
    embedding = embed_text(models["embed"], extracted_text).reshape(1, config.llm.embedding.dim)
    prediction = models["class"].predict(embedding)

    # TODO remove conversion/work with batched output
    return str(prediction[0])


def classify(config, extracted_text, models=None):
    match config.llm.method:
        case "prompt":
            return classify_prompt(config, extracted_text)
        case "embedding":
            return classify_embedding(config, extracted_text, models)
        case _:
            raise Exception(f"Unspecified method for classification: {config.llm.method}")
