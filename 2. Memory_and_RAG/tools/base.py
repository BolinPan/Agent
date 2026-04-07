from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """
    Define tool parameter
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """
    Define tool base class - All tools should inherit from this class and implement the abstract methods
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        Execute tool
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        Get tool parameter definitions"""
        pass
    

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters
        """
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.dict() for param in self.get_parameters()]
        }
    

    def __str__(self) -> str:
        """
        String representation of the tool
        """
        return f"Tool(name={self.name})"
    

    def __repr__(self) -> str:
        """
        Official string representation of the tool
        """
        return self.__str__()