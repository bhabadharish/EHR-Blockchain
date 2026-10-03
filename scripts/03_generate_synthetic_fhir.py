import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.synthetic_fhir_generator import generate_fhir_splits

def main():
    print("=" * 70)
    print("STAGE 3: GENERATING SYNTHETIC FHIR / EHR SECURITY EVENT DATASET")
    print("=" * 70)
    generate_fhir_splits(output_dir="data/raw/synthetic_fhir")

if __name__ == "__main__":
    main()
