from typing import Dict, Any, Optional


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'overview')
      - parameters    (a dictionary of parameter_name -> value)
    """

    def __init__(self, function_name: str, parameters: Dict[str, Any], function_id: Optional[str] = None):
        self.function_name = function_name
        self.parameters = parameters
        self.function_id = function_id

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, object]:
        return {"function": self.function_name, "parameters": self.parameters, "id": self.function_id}