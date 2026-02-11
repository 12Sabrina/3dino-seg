import json
import argparse
from pathlib import Path

DEFAULT_MODALITY_ORDER = [
    "T1-weighted",
    "T1-weighted Contrast Enhanced",
    "T2-weighted",
    "T2-weighted FLAIR",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label-key", default="Segmentation")
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITY_ORDER)
    args = parser.parse_args()

    def process(path):
        if not path or not path.exists():
            return []
        with open(path) as f:
            payload = json.load(f)
            data = payload.get("data", payload.get("training", [])) # Handle both formats
        
        out = []
        for item in data:
            modalities = item.get("modalities", {})
            image_paths = []
            for mod in args.modalities:
                if mod in modalities:
                    image_paths.append(modalities[mod])
            
            # Create a copy to preserve all top-level attributes (Survival_days, subtypes, etc.)
            new_item = item.copy()
            new_item["image"] = image_paths
            # For segmentation, the 'label' key in datalist should be the path to nii.gz
            if args.label_key in modalities:
                new_item["label"] = modalities[args.label_key]
            elif "label" not in new_item:
                new_item["label"] = ""
            
            out.append(new_item)
        return out

    train_entries = process(args.train_json)
    val_entries = process(args.val_json)
    
    result = {
        "training": train_entries,
        "validation": val_entries,
        "test": val_entries # Default to val for test
    }
    
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done: {args.output_json}")

if __name__ == "__main__":
    main()
