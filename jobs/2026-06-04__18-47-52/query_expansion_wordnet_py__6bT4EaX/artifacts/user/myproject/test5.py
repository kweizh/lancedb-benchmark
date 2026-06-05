from nltk.corpus import wordnet as wn
token = "car"
synonyms = []
for synset in wn.synsets(token):
    for lemma in synset.lemmas():
        name = lemma.name().lower()
        if "_" not in name and " " not in name and name != token and name not in synonyms:
            synonyms.append(name)
print(synonyms[:3])
