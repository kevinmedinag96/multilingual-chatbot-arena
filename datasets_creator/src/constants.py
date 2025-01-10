import pathlib
PROMPT_TEMPLATE = """
You are an expert in assesing LLM's model response based on a prompt. I will give you an input prompt (**prompt**) with two different responses coming from fellow LLM models; the first model's response is called **response_a** and second model's response is **response_b**. You can find the previous information after the double slashes (//), respecting the correct title based on the proper input.Your task is to assess the content of each response based on its quality and human's language similarity, then choose the model's response which adheres best to the given guidelines.\nYour response must obey the following format: 'Best model is model_[] based on its human preferability response for the input prompt.'. You will substitute '[]' with either 'a' if you think **response_a** is better than **response_b**, or 'b' otherwise.\n\n//\n**prompt**:\n{prompt}\n\n**response_a**:\n{response_a}\n\n**response_b**:\n{response_b}
"""
CURR_PATH: str =  pathlib.Path().resolve().__str__()
SLASH = "/"
UNDERSCORE = "_"