def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    words=corpus.split()
    w = len(words)
    s=0
    for item in words:
        if item==word:
            s+=1
    return s/w