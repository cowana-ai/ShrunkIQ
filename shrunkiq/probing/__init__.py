import nltk


def ensure_nltk_resources():
    resources = [
        'punkt',
        'wordnet',
        'averaged_perceptron_tagger',  # Change to the correct resource name
        'universal_tagset',
        'stopwords',
        'averaged_perceptron_tagger_eng',
        "punkt_tab"
    ]

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')  # Check for tokenizer resources
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')  # Check for corpora resources (like wordnet)
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{resource}')  # Check for tagger resources
                except LookupError:
                    print(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource)

# Call this function in your package's main module or entry point
ensure_nltk_resources()
