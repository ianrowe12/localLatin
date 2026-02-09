from datasets import load_dataset
try:
    ds = load_dataset("musts/english", split="test")
    print(f"Count: {len(ds)}")
except Exception as e:
    print(f"Error: {e}")
