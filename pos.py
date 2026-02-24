import spacy
from lemminflect import getInflection

nlp = spacy.load("en_core_web_sm")

sentence = "She was running fast."
doc = nlp(sentence)

for token in doc:
    if token.text == "running":
        original_pos = token.tag_   # VBG
        
        candidate = "jog"
        inflected = getInflection(candidate, tag=original_pos)
        
        print(inflected)