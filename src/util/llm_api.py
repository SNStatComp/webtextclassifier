import openai


def LLM_API(config, variable, message_content=""):

    client = openai.OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url
    )

    prompt_content = f"{message_content}, does the above-mentioned text, relating to a web-page, contain text with its main content about: {variable}?"

    response = client.chat.completions.create(
        model = config.llm.model,
        messages = [
            {"role":"system", "max_completion_tokens":"1", "content":"You are an expert at classifying texts, your answer is a one-word response, chosen from [Yes, No]."},
            {"role":"user", "content": prompt_content},
        ],
        temperature = 0.0,
        max_completion_tokens = 1,
        max_tokens = 1
    )

    return response.choices[0].message.content
