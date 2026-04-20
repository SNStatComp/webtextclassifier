import openai


# Method to create openai api client
def create_client(config):
    client = openai.OpenAI(
        api_key=config.llm.prompt.api_key,
        base_url=config.llm.prompt.api_url
    )

    return client


# Method to prompt openai api model
def prompt_LLM(config, client, extracted_text="", variable=None):

    prompt_content = f"{extracted_text}, Would you say that the above-given text that was taken from a web-page relates mainly in its content to: {variable}? Reply only with either Yes or No"

    response = client.chat.completions.create(
        model = config.llm.prompt.model,
        messages = [
            {
                "role":"system",
                "max_completion_tokens":"1",
                "content":"You are an expert at classifying texts, your answer is a one-word response, chosen from [Yes, No]."},
            {
                "role":"user",
                "content": prompt_content
            },
        ],
        temperature = 0.0,
        max_completion_tokens = 1,
    )

    return response.choices[0].message.content
