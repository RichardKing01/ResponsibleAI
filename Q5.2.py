import numpy as np
from gensim.models import KeyedVectors

print("Hello")

Input_File = open("questions-words.txt", "r", encoding="utf-8")
i = 10
j = 0

model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def check(words: list):
    w1, w2, w3, w4 = words

    # Skip if any word not in vocabulary
    if any(word not in model for word in [w1, w2, w3, w4]):
        print("Skipped: not in vocabulary")
        return

    vec = model[w2] - model[w1] + model[w3]

    similarity = cosine_similarity(vec, model[w4])

    print(f"Similarity between predicted and actual word is: {similarity}")

while j != i:
    sentence = Input_File.readline()

    if sentence == "":
        break

    if sentence.startswith(":"):
        continue

    print(sentence.strip())

    words = sentence.strip().split()

    if len(words) == 4:
        check(words)
        j += 1

Input_File.close()
