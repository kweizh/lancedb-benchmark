import nltk
from nltk.corpus import wordnet as wn

def get_synonyms(token: str, max_synonyms: int = 3) -> list[str]:
    token = token.lower()
    synonyms = set()
    for synset in wn.synsets(token):
        for lemma in synset.lemmas():
            syn_name = lemma.name().lower()
            if syn_name != token and "_" not in syn_name and "-" not in syn_name:
                synonyms.add(syn_name)
                if len(synonyms) >= max_synonyms:
                    return list(synonyms)
    return list(synonyms)

test_tokens = ["car", "fast", "happy"]
for t in test_tokens:
    print(f"{t}: {get_synonyms(t)}")
