import transformers
from datasets import load_dataset

dataset = load_dataset("newsqa", cache_dir=".cache/")