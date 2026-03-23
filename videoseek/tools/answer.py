from videoseek.utils import call_llm_api


answer_tool = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Based on the given trajectory, generate the final answer to the question.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def execute_answer(config: dict, parameters: dict) -> str:
    """
    Execute the answer tool.
    """
    question = parameters['question']
    messages = parameters['messages']
    messages.append({
        "role": "user",
        "content": (
            f"Question:\n{question}\n\n"
            "Please directly provide the final answer."
        ),
    })
    response = call_llm_api(
        messages=messages,
        model_name=config['model_name'],
        api_base=config['api_base'],
        api_key=config['api_key'],
        api_version=config['api_version'],
        max_tokens=config['max_tokens'],
        reasoning_effort=config['reasoning_effort'],
        seed=config['seed'],
        temperature=config['temperature'])
    
    return response.choices[0].message.content