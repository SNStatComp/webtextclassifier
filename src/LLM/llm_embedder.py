from sentence_transformers import SentenceTransformer


def create_model(config):
    model = SentenceTransformer(config.llm.embedding.model)
    return model


def embed_text(model, text, variable):
    prompt = f"{text}, does the above-mentioned text, relating to a web-page, contain text with its main content about: {variable}?"
    return model.encode(prompt)
