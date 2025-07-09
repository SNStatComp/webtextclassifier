import requests
import justext


def extract_text(url, country):
    try:
        response = requests.get(url, timeout=5)
    except:
        raise Exception(f"Could not request url: {url}")

    if not (response.status_code >= 200 and response.status_code < 400):
        return ""

    # TODO determine language by text content rather than domain
    language = "English"
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
