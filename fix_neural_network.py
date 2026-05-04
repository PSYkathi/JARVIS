import requests, re

headers = {'User-Agent': 'Mozilla/5.0'}

# Using Wikipedia's REST API instead of the query API
articles = {
    'Artificial_neural_network': 'data/raw/neural_network.txt',
    'Reasoning':                  'data/raw/reasoning.txt',
    'Logic':                      'data/raw/logic_extra.txt',
}

for title, raw_path in articles.items():
    try:
        url = f'https://en.wikipedia.org/api/rest_v1/page/plain/{title}'
        r = requests.get(url, headers=headers, timeout=30)
        text = r.text
        if len(text) < 1000:
            print(f"❌ {title}: too short ({len(text)} chars)")
            continue
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(text)
        cleaned = re.sub(r'\n{3,}', '\n\n', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        clean_path = raw_path.replace('raw', 'clean')
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"✅ {title}: {len(text):,} chars")
    except Exception as e:
        print(f"❌ {title}: {e}")

print("Done!")