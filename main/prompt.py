from dataclasses import dataclass

@dataclass
class Action:
    thought: str
    bash: str

@dataclass
class Step:
    action: Action
    observation: str