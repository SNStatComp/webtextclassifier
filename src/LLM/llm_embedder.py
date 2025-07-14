from sentence_transformers import SentenceTransformer


def create_model(config):
    model = SentenceTransformer(config.llm.model)
    return model


def embed_text(model, text):
    return model.encode(text)
