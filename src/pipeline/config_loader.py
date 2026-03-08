import yaml
import os
from pathlib import Path
from typing import Any, Dict

class ConfigNode(dict):
  def __getattr__(self,key):
    try:
      val = self[key]
      return ConfigNode(val) if isinstance(val, dict) else val
    except KeyError:
      raise AttributeError(f"Config has no attribute '{key}'")

class ConfigLoader:
  _instance = None
  
  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(ConfigLoader, cls).__new__(cls)
    return cls._instance
  
  def __init__(self, config_dir: str = "configs"):
    if hasattr(self, "initialized"):
      return
    
    self.config_dir = Path(config_dir)
    if not self.config_dir.exists():
      raise FileNotFoundError(f"Configuration Directory {config_dir} not found.")
    
    self._raw_configs = {
      "pipeline": self._load_file("pipeline.yaml"),
      "preprocessing": self._load_file("preprocessing.yaml"),
      "models": self._load_file("models.yaml"),
      "evaluation": self._load_file("evaluation.yaml"),
      "tiling": self._load_file("tiling.yaml")
    } 
    
    self.pipeline = ConfigNode(self._raw_configs["pipeline"])
    self.preprocessing = ConfigNode(self._raw_configs["preprocessing"])
    self.models = ConfigNode(self._raw_configs["models"])
    self.evaluation = ConfigNode(self._raw_configs["evaluation"])
    self.tiling = ConfigNode(self._raw_configs["tiling"])
    
    self.initialized = True
    print(f"Successfully Loaded Neuro-Symbolic HTR configuration from {self.config_dir}")
    
  def _load_file(self, filename: str) -> Dict[str, Any]:
    file_path = self.config_dir / filename
    if not file_path.exists():
      print(f"Warning: {filename} missing. Initializing with empty config.")
      return {}
    
    with open(file_path, "r") as f:
      return yaml.safe_load(f) or {}
    
  def get_all(self) -> Dict[str, Any]:
    return self._raw_configs

# Usage Example:
# config = ConfigLoader()
# print(config.models.detection.confidence_threshold)