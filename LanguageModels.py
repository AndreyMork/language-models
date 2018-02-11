import math
import re


def tokenize(text):
    return re.findall('\w+', text.lower())


def load_text(file_name):
    with open(file_name) as f:
        return tokenize(f.read())


def get_ngram_counts(words, n):
    """Пройдя по words (списку токенов без знаков препинания), функция возвращает словарь вида {ngram: count} для всех n-граммов, а также n-граммов низшего порядка.
    Например, get_ngram_counts(words, 3) возвращает словарь частотности всех триграммов, биграммов и юниграммов, считая их по списку words.
    n-граммы лучше представлять в виде кортежей, например: ('слово1', 'слово2', 'слово3')"""

    counts = dict()
    for i in range(1, n + 1):
        for j in range(0, len(words) - i + 1):
            temp = tuple(words[j: j + i])
            counts[temp] = counts.get(temp, 0) + 1
    return counts


def get_prob(text, ngrams, n, corp_size, logspace=False):
    """Возвращает вероятность строки text, основываясь на словаре ngrams и
    оценивая параметры модели с помощью максимального правдоподобия.
    n - длина n-граммов в модели (напр. 1, 2 или 3),
    corp_size - количество токенов в корпусе
    (для оценки параметров юниграмм-модели)."""

    if logspace:
        prob = 0
    else:
        prob = 1
    text = text.split()

    if len(text) < n:
        return None

    if n == 1:
        for w in text:
            if ngrams.get(tuple(w.split()), 0) == 0:
                return None

            if logspace:
                prob += math.log(ngrams[tuple(w.split())] / corp_size, 2)
            else:
                prob *= ngrams[tuple(w.split())] / corp_size

    else:
        for i in range(0, len(text) - n + 1):
            temp = tuple(text[i: i + n])
            if ngrams.get(temp, 0) == 0:
                return None

            if logspace:
                prob += math.log(ngrams[temp] / ngrams[temp[:n - 1]], 2)
            else:
                prob *= ngrams[temp] / ngrams[temp[:n - 1]]

    if logspace:
        return 2 ** prob
    else:
        return prob


def get_prob_sm(text, ngrams, n, corp_size, logspace=False):
    """Возвращает вероятность строки text, основываясь на словаре ngrams и
    оценивая параметры модели с помощью максимального правдоподобия.
    n - длина n-граммов в модели (напр. 1, 2 или 3),
    corp_size - количество токенов в корпусе
    (для оценки параметров юниграмм-модели)."""

    if logspace:
        prob = 0
    else:
        prob = 1
    text = text.split()

    if len(text) < n:
        return None

    v = len([k for k in ngrams if len(k) == 1])
    if n == 1:
        for w in text:
            if logspace:
                prob += math.log((ngrams.get(tuple(w.split()), 0) + 1) / (corp_size + v), 2)
            else:
                prob *= (ngrams.get(tuple(w.split()), 0) + 1) / (corp_size + v)

    else:
        for i in range(0, len(text) - n + 1):
            temp = tuple(text[i: i + n])
            if logspace:
                prob += math.log(ngrams.get(temp, 0) / (ngrams.get(temp[:n - 1])), 2)
            else:
                prob *= (ngrams.get(temp, 0) + 1) / (ngrams.get(temp[:n - 1], 0) + v)
    if logspace:
        return 2 ** prob
    else:
        return prob



# words = load_text("corpus.txt")
corpus = 'I saw a cat and a dog. The cat was sleeping, and the dog was awake. I woke up the cat.'
words = tokenize(corpus)

corp_size = len(words)
ngrams = get_ngram_counts(words, 3)

tests = ('a cat', 'the cat', 'the dog', 'the woke', 'the cat was awake', 'as a result of the explosion')

t = "woke up the cat"
for n in range(1, 4):
    print(t, get_prob(t, ngrams, n, corp_size), "   ", n)
    print(t, get_prob_sm(t, ngrams, n, corp_size), "   ", n, "sm")
    print(t, get_prob(t, ngrams, n, corp_size, logspace=True), "   ", n, "log")
    print(t, get_prob_sm(t, ngrams, n, corp_size, logspace=True), "   ", n, "log sm")