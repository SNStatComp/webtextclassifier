from .llm_prompter import prompt_LLM
from .llm_embedder import embed_text
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib


def create_classification_model(config, pretrained=True):
    match config.llm.embedding.classifier:
        case "DecisionTree":
            model = DecisionTreeClassifier()
            if pretrained:
                # We use joblib for loading pkls
                model = joblib.load(config.llm.embedding.classifier_path)
                print(f"Loaded weights for {config.llm.embedding.classifier} classifier")
            return model
        case "SVM_rbf":
            model = SVC(kernel="rbf")
            if pretrained:
                # We use joblib for loading pkls
                model = joblib.load(config.llm.embedding.classifier_path)
                print(f"Loaded weights for {config.llm.embedding.classifier} classifier")
            return model
        case _:
            warnings.warn(f"TODO! {config.llm.embedding.classifier} not yet supported")


def classify_prompt(config, extracted_text, variable):
    # For now, just prompt the llm
    return 1 if prompt_LLM(config, extracted_text, variable) == "Yes" else 0


# TODO restructure code to process in batches
def classify_embedding(config, extracted_text, models):
    embedding = embed_text(models["embedding"], extracted_text)
    prediction = models["classification"].predict(embedding)
    return prediction.astype(float).astype(int).astype(str)
