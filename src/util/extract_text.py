import requests
import justext


def extract_text(config, url, country):
    try:
        response = requests.get(url, timeout=config.requests.timeout)
    except Exception as e:
        raise Exception(f"Could not request url: {url}, error: {e}")

    if not (response.status_code >= 200 and response.status_code < 400):
        return ""

    # TODO determine language by text content rather than domain
    language = config.requests.default_language
    if country == "DE":
        language = "German"
    elif country == "NL":
        language = "Dutch"
    elif country == "PL":
        language = "Polish"
    elif country == "AT":
        language = "German"

    # TODO right now we extract text for every variable, but only needs to be done once
    extracted_paragraphs = justext.justext(response.content, justext.get_stoplist(language))
    extracted_text = "".join([paragraph.text for paragraph in extracted_paragraphs if (paragraph.cf_class in ["good", "neargood"] or paragraph.class_type in ["good", "neargood"])])

    return extracted_text
