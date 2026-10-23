import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import scientific_runtime
scientific_runtime()
