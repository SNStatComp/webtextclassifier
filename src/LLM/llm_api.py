import openai


def prompt_LLM(config, extracted_text="", variable=None):
    client = openai.OpenAI(
        api_key=config.llm.prompt.api_key,
        base_url=config.llm.prompt.url
    )

    prompt_content = f"{extracted_text}, does the above-mentioned text, relating to a web-page, contain text with its main content about: {variable}?"

    response = client.chat.completions.create(
        model = config.llm.prompt.model,
        messages = [
            {"role":"system", "max_completion_tokens":"1", "content":"You are an expert at classifying texts, your answer is a one-word response, chosen from [Yes, No]."},
            {"role":"user", "content": prompt_content},
        ],
        temperature = 0.0,
        max_completion_tokens = 1,
        max_tokens = 1
    )

    return response.choices[0].message.content
