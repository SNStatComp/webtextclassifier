from sentence_transformers import SentenceTransformer


def embed_sentence_transformer(config, text):
    # TODO can probably make this more efficient by instantiating model once/passing text in parallel
    model = SentenceTransformer(config.llm.model)

    return model.encode(text)
