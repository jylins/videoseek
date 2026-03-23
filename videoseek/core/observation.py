from .action import Action


class Observation:
    """
    Represents an observation with:
      - action      (the Action that produced this observation)
      - output      (the output of the action)
    """

    def __init__(self, action: Action, outcome: str):
        self.action = action
        self.outcome = outcome

    def __str__(self) -> str:
        func_name = getattr(self.action, "function_name", "") if self.action else ""
        parameters = getattr(self.action, "parameters", {}) if self.action else {}

        if not func_name:
            return "There is no function call in your response. YOU MUST USE A FUNCTION CALL IN EACH RESPONSE."
        parameters_str = ", ".join([f"{k}='{v}'" for k, v in parameters.items()])
        return f"Execution output of \\[{func_name}({parameters_str})]:\n{self.outcome}"

