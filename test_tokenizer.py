from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer('data/tokenizer/vocab.json', 'data/tokenizer/merges.txt')
tests = ['Hello, I am JARVIS', 'artificial intelligence', 'neural network training', 'deep learning model']

for t in tests:
    enc = tok.encode(t)
    print(f'Input : {t}')
    print(f'Tokens: {enc.tokens}')
    print(f'IDs   : {enc.ids}')
    print()