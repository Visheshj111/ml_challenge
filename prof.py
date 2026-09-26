import sys
import cProfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution" / "src"))
from features import extract_features

s1 = ["Test Business 0 LLC"] * 50000
s2 = ["Test Business 1 LLC"] * 50000

def run():
    for i in range(50000):
        extract_features(s1[i], s1[i], "US", s1[i], s2[i], s2[i], "US", s2[i])

cProfile.run("run()", sort="cumtime")
