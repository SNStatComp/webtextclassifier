from .llm_api import prompt_LLM
from .llm_embedder import embed_text


def classify_prompt(config, extracted_text, variable):
    # For now, just prompt the llm
    return 1 if prompt_LLM(config, extracted_text, variable) == "Yes" else 0


# TODO restructure code to process in batches
def classify_embedding(config, extracted_text, variable, models):
    embedding = embed_text(models["embed"], extracted_text, variable).reshape(1, config.llm.embedding.dim)
    prediction = models["class"].predict(embedding)

    # TODO remove conversion/work with batched output
    return str(prediction[0])


def classify(config, extracted_text, variable, models=None):
    match config.llm.method:
        case "prompt":
            return classify_prompt(config, extracted_text, variable)
        case "embedding":
            return classify_embedding(config, extracted_text, variable, models)
        case _:
            raise Exception(f"Unspecified method for classification: {config.llm.method}")
