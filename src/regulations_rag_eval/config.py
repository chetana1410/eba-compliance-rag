import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / "shared.env", override=False)
load_dotenv(PROJECT_ROOT / "local.env", override=True)


PATH_KEYS = ["DATA_PATH"]

# Step 4: Normalize those env vars to absolute paths
for key in PATH_KEYS:
    value = os.environ.get(key)
    if value:
        path = Path(value)
        #print(f"Normalizing {key}: {path}")
        if not path.is_absolute():
            abs_path = str(PROJECT_ROOT / path)
            os.environ[key] = abs_path  # Overwrite with absolute path
            #print(f"Normalized {key} to absolute path: {abs_path}")

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
