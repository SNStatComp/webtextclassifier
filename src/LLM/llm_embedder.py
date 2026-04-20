from sentence_transformers import SentenceTransformer


# Create model object for embedding model
def create_embedding_model(config, local=False):
    print("Loading SentenceTransformer with weights...")

    if (local):
        print(f"Using local embedding model {config.llm.embedding.model} at path: {config.llm.embedding.model_weights}")
        model = SentenceTransformer(config.llm.embedding.model_weights)
    else:
        model = SentenceTransformer(config.llm.embedding.model)

    print("Model device:", model.device)
    return model


# Create text embedding from given model
def embed_text(model, text):
    return model.encode(text)
