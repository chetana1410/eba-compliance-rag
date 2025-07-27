import os
from pathlib import Path

def implementation_system_path(implementation_name: str) -> str:
    base_path = os.environ.get("DATA_PATH")
    if not base_path:
        raise ValueError("DATA_PATH environment variable not set")
    
    system_path = Path(base_path) / implementation_name / "system"
    
    if not system_path.exists():
        system_path.mkdir(parents=True, exist_ok=True)
    
    return str(system_path)