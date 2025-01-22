import pathlib

SYSTEM_TEMPLATE = 'You are a specialist in evaluating multilingual chat responses, with a focus on comparing and ranking outputs from different LLMs. Your primary goal is to determine which response is more likely to be preferred by humans based on factors such as clarity, relevance, tone, and overall quality.\n'
PROMPT_TEMPLATE = """Below is a prompt with two possible responses (**Response A** and **Response B**). Evaluate them, select the best one and answer in the following format (it is imperative that you respect the specified format, do not add any more text than what I ask for):\n1.- Write 'model_a' if the **Response A** is better than **Response B**, otherwise write 'model_b'.\n\n**Prompt**:\n{prompt}\n\n**Response A**:\n{response_a}\n\n**Response B**:\n{response_b}\n"""
CURR_PATH: str =  pathlib.Path().resolve().__str__()
SLASH = "/"
UNDERSCORE = "_"