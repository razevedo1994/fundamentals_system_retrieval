import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    documents = [
        "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
        "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
        "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
        "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
        "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
        "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
        "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
        "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
        "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
        "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
        "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
    ]

    def preprocess(input: list[str]):
        text_lower = input.lower()

        tokens = nltk.word_tokenize(text_lower)

        return [word for word in tokens if word.isalnum()]

    preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
    print(preprocessed_docs)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
    print(tfidf_matrix)

    query = "machine learning"
    query_vector = vectorizer.transform([query])
    print(query_vector)

    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    print(similarities)
