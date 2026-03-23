from typing import List, Dict, Any
from .action import Action
from .observation import Observation


class TrajectoryStep:
    """A single step in the agent trajectory."""

    def __init__(self, step_id: int, thought: str, action: Action, observation: Observation):
        self.step_id = step_id
        self.thought = thought
        self.action = action
        self.observation = observation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action.to_dict(),
            "observation": str(self.observation),
        }


class Trajectory:
    """Complete trajectory of an agent run."""
    def __init__(self, question: str, steps: List[TrajectoryStep], final_answer: str, finish_reason: str):
        self.question = question
        self.steps = steps
        self.final_answer = final_answer
        self.finish_reason = finish_reason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": max(s.step_id for s in self.steps),
            "final_answer": self.final_answer,
            "finish_reason": self.finish_reason,
        }