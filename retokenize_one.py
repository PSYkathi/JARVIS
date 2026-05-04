from tokenizers import ByteLevelBPETokenizer
import os

tok = ByteLevelBPETokenizer('data/tokenizer/vocab.json', 'data/tokenizer/merges.txt')

files = {
    'data/clean/logic_extra.txt':  'data/tokenized/logic_extra.ids',
}

for clean_path, out_path in files.items():
    if not os.path.exists(clean_path):
        print(f"❌ Not found: {clean_path}")
        continue
    with open(clean_path, encoding='utf-8') as f:
        text = f.read()
    ids = tok.encode(text).ids
    with open(out_path, 'w') as f:
        f.write(' '.join(map(str, ids)))
    print(f"✅ {clean_path}: {len(ids):,} tokens")