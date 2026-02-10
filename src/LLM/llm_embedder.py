from sentence_transformers import SentenceTransformer


def create_embedding_model(config):
    # TODO loading weights from dir
    print("Loading SentenceTransformer with weights...")
    model = SentenceTransformer(config.llm.embedding.model)
    return model


def embed_text(model, text):
    return model.encode(text)
