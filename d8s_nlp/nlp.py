def word_cloud(text, output_file_path=None):
    """Create a word cloud based on the given text."""
    import matplotlib.pyplot as plt
    from wordcloud import STOPWORDS, WordCloud

    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(text)

    figure, axes = plt.subplots()

    axes.axis("off")
    axes.imshow(wordcloud)
    plt.show()

    # TODO: it would be nice to save the image of the word cloud
    if output_file_path:
        figure.savefig(output_file_path)
        print('Saved an image of this word-cloud to "{}"'.format(output_file_path))

    return plt


def word_stem(word, stemmer='porter'):
    """Return the stem of the given word."""
    import nltk

    available_stemmers = ['porter', 'lancaster']
    if stemmer.lower() not in available_stemmers:
        print('! Invalid stemmer given: {}\nAvailable stemmers are: {}'.format(stemmer, available_stemmers))
        return

    stemmer_object = eval('nltk.{}Stemmer()'.format(stemmer.title()))
    return stemmer_object.stem(word)


def words_generate(letters_list, min_word_length=2, required_characters_list=None):
    """Generate all possible, valid words from the given list of letters."""
    from d8s_lists import iterableNotIn

    valid_word_list = nltk_word_list()
    letters = frequency_distribution(''.join(letters_list))

    # TODO: I updated this function to use frequency_distribution rather than nltk.freqDist... make sure this is working properly
    valid_words = [
        word for word in valid_word_list if len(word) >= min_word_length and frequency_distribution(word) <= letters
    ]

    if required_characters_list is not None:
        # make sure the required characters are present
        # TODO: write a function for filtering
        valid_words = list(filter(lambda x: iterableNotIn(required_characters_list, x) == [], valid_words))

    return valid_words


def word_search(text, word):
    """Search for all instances of the given word in the given text."""
    t = nltk_text(text)
    return t.concordance(word)


def word_syn_sets(word):
    """Return the synonym sets for the given word."""
    from nltk.corpus import wordnet as wn

    return wn.synsets(word)


def word_synonyms_common(word):
    """Return the synonyms for the most common meaning of the given word."""
    return word_syn_sets(word)[0].lemma_names()


def word_synonyms(word):
    """Return the synonyms for all meanings of the given word."""
    synonyms = []
    for synset in word_syn_sets(word):
        synonyms.append(synset.lemma_names())
    return synonyms


def word_similar_words(word):
    """Find words that are similar to the given word."""
    # TODO: implement!
    raise NotImplementedError


def word_definition_common(word):
    """Return the possible definitions of the given word."""
    return word_syn_sets(word)[0].definition()


def word_use_in_sentence(word):
    return word_syn_sets(word)[0].examples()


def text_valid_english_words(text: str):
    """Return the number of valid, English words in the given text."""
    from words_module import wordIsValidEnglishWord

    words_in_text = words(text)
    valid_english_words = []

    for word in words_in_text:
        if wordIsValidEnglishWord(word):
            valid_english_words.append(word)

    return valid_english_words


def text_valid_english_word_count(text: str) -> int:
    """Return the number of valid, English words in the given text."""
    valid_english_words = text_valid_english_words(text)
    valid_english_word_count = len(valid_english_words)
    return valid_english_word_count


def word_definitions(word):
    """Return the most common definition of the given word."""
    # TODO: this function could also be named "define"
    # TODO: also return the part of speach with each definition
    definitions = []
    for synset in word_syn_sets(word):
        definitions.append(synset.definition())
    return definitions


def word_hyponyms_common(word):
    """Return the hyponyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) of the most common meaning of the given word."""
    return [a.lemma_names() for a in word_syn_sets(word)[0].hyponyms()]


def word_hyponyms(word):
    """Return the hyponyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) for all meanings of the given word."""
    hyponyms = []
    for synset in word_syn_sets(word):
        hyponyms.append([a.lemma_names() for a in synset.hyponyms()])
    return hyponyms


def word_root_hypernym(word):
    """Return the root hypernym (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy)."""
    return [a.lemma_names() for a in word_syn_sets(word)[0].root_hypernyms()]


def word_hypernyms_common(word):
    """Return the hypernyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) of the most common meaning of the given word."""
    return [a.lemma_names() for a in word_syn_sets(word)[0].hypernyms()]


def nltk_text(text):
    """Return nltk.text.Text for the given text."""
    import nltk.text

    return nltk.text.Text(text)


def word_hypernyms(word):
    """Return the hypernyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) for all meanings of the given word."""
    hypernyms = []
    for synset in word_syn_sets(word):
        hypernyms.append([a.lemma_names() for a in synset.hypernyms()])
    return hypernyms


def text_tokens(text):
    """Return the tokens for the given text."""
    import nltk

    # TODO: this function could also be called "tokenize"
    return nltk.word_tokenize(text)


def frequency_distribution(text):
    # TODO: is there a better way to get a freqdist that doesn't involve import nltk.book?
    from nltk.book import FreqDist

    return FreqDist(text)


def word_dispersion_plot(text, word_list):
    t = nltk_text(text)
    # TODO: write a function to do this
    # make sure that the word_list is actually a list
    if isinstance(word_list, str):
        word_list = list(word_list)
    t.dispersion_plot(word_list)


def word_frequency(text):
    from d8s_lists import count

    word_list = words(text)
    return count(word_list)


def similar_words(text, word):
    """Find words which are used in a similar context as the given word."""
    t = nltk_text(text)
    t.similar(word)


# TODO: this can also be called "lexical richness" or "lexical_diversity" (see http://www.nltk.org/book/ch01.html) - we should capture that somewhere
def word_repitition(text):
    """Return the ratio of the number of unique words with the total number of words in the text."""

    unique_word_count = len(words_unique(text)) - 1
    word_count = len(words(text))
    # todo: the construct below is very similar to the stuff in the `sentence_average_length` function... could consolidate
    if word_count > 0:
        return 1 - (unique_word_count / word_count)
    else:
        print('No words found in the given text')
        return None


def word_common_contexts(text, words_list):
    t = nltk_text(text)
    return t.common_contexts(words_list)


def text_plot_frequency(text, limit=50, cumulative=False):
    # TODO: it would be cool to be able to print each of the most common words as a percent of the total words
    frequency_distribution(text).plot(limit, cumulative=cumulative)


def text_tabulate_frequency(text, limit=50, cumulative=False):
    """."""
    return frequency_distribution(text).tabulate(limit, cumulative=cumulative)


def text_hapaxes(text):
    """Return the hapaxes (the words that only occur once) in the given text."""
    hapaxes = frequency_distribution(text).hapaxes()
    return hapaxes


def text_collocations(text):
    t = nltk_text(text)
    # this function is weird in that it prints out the collocations, but doesn't return them
    t.collocations()


def word_repitition_percent(text):
    """Return the percentage of the words which are repeated in the text."""
    from d8s_math import percent

    word_repitition_ratio = word_repitition(text)
    return percent(word_repitition_ratio)


def text_tags(text):
    """Return each word in the text tagged with its part of speech."""
    # there is a helpful guide to the tags used by nltk here: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
    blob = text_blob(text)
    return blob.tags


# TODO: write a function to get the part of speech from a word


def _text_tag_filter(tags, tag_filter):
    """Filter the tags and return words whose tags match the given tag_filter."""
    return [tag for tag in tags if tag[1].startswith(tag_filter)]


def _text_tag_deduplication(tags):
    """Deduplicate the tags and return only the name and not the part of speech."""
    from d8_lists import deduplicate

    return deduplicate([tag[0] for tag in tags])


def text_nouns(text):
    """Get all nouns from the text."""
    tags = text_tags(text)
    return _text_tag_deduplication(_text_tag_filter(tags, 'NN'))


def text_verbs(text):
    """Get all nouns from the text."""
    tags = text_tags(text)
    return _text_tag_deduplication(_text_tag_filter(tags, 'VB'))


def proper_nouns(text):
    """Get all of the proper nouns in text."""
    tags = text_tags(text)
    return _text_tag_deduplication(_text_tag_filter(tags, 'NNP'))


def proper_nouns_count(text):
    """."""
    from d8s_lists import count

    tags = text_tags(text)
    proper_nouns = [tag[0] for tag in _text_tag_filter(tags, 'NNP')]
    return count(proper_nouns)


def words_count(text):
    """Return the number of words in the given text."""
    return len(words(text))


def word_count(text, word, ignore_case=True):
    """Find the count of the given word in the text."""
    from d8s_strings import lowercase

    text_words = words(text)

    if ignore_case:
        text_words = lowercase(text_words)
        word = word.lower()

    return text_words.count(word)


def text_contains(text, word, ignore_case=True):
    """Return whether or not the text contains the given word."""
    from d8s_strings import lowercase

    if ignore_case:
        text = lowercase(text)
        word = word.lower()

    return word in text


def tfidf(word, text, multiple_texts):
    """Find the "Term Frequency, Inverse Document Frequency" for the given word in the given text using the multiple texts."""
    import math

    tf = word_count(text, word) / words_count(text)
    idf = math.log(len(multiple_texts) / (1 + sum([1 for t in multiple_texts if text_contains(t, word)])))

    return tf * idf


def text_skeleton(text):
    """Return the verbs and nouns in the text."""
    from d8s_lists import iterableCombine

    verbs = text_verbs(text)
    nouns = text_nouns(text)
    return iterableCombine(nouns, verbs)


def words_unique(text):
    """Get a deduplicated list of all of the words in the given text."""
    from d8s_lists import deduplicate
    from d8s_strings import lowercase

    word_list = lowercase(words(text))
    unique_words = deduplicate(word_list)
    return unique_words


def subjectivity(string):
    ment = sentiment(string)
    return ment.subjectivity


def subjectivity_number_line(string):
    from d8s_math import numberLine

    return numberLine(subjectivity(string), 0, 1, 0.1)


def nltk_word_list():
    """Return the nltk wordlist."""
    from nltk.corpus import words

    return words.words()


# TODO: this function should be able to remove stopwords from both a list and a string
def stopwords_remove(string):
    from d8s_lists import iterableNotIn
    from d8s_strings import lowercase
    from wordcloud import STOPWORDS

    # not sure if lowercasing this is the correct move, but if this is not done, words like "And" and "AND" will not be removed
    word_list = words(lowercase(string))
    # use nltk's stopwords
    non_stop_words = iterableNotIn(word_list, nltk_stopwords_list())
    # use wordcloud's stopwords
    non_stop_words = iterableNotIn(non_stop_words, STOPWORDS)
    return ' '.join(non_stop_words)


def sentence_average_length(string):
    """Return the average length of a sentence in the string."""

    word_count = len(words(string))
    sentence_count = len(sentences(string))
    if sentence_count > 0:
        return word_count / sentence_count
    else:
        print('No sentences found in the given string')
        return None


def text_blob(string):
    """Return a textblob for the given string."""
    from textblob import TextBlob

    return TextBlob(string)


def sentences(string):
    blob = text_blob(string)
    return blob.sentences


def ngrams(string, n=3):
    blob = text_blob(string)
    # join the ngrams into strings
    grams = [' '.join(gram) for gram in blob.ngrams(n)]
    return grams


def ngrams_common(string, n=3):
    from d8s_lists import count

    grams = ngrams(string, n)
    if grams:
        sorted_grams = count(grams)
    else:
        sorted_grams = {}
    return sorted_grams


def noun_phrases(string):
    blob = text_blob(string)
    phrases = [phrase for phrase in blob.noun_phrases]
    return phrases


def noun_phrases_common(string):
    from d8s_lists import count

    phrases = noun_phrases(string)
    return count(phrases)


def polarity(string):
    ment = sentiment(string)
    return ment.polarity


def polarity_number_line(string):
    from d8s_math import numberLine

    return numberLine(polarity(string), -1, 1, 0.1)


def words(text):
    blob = text_blob(text)
    return blob.words


def correct_spelling(string):
    blob = text_blob(string)
    return blob.correct()


def sentiment(string):
    blob = text_blob(string)
    return blob.sentiment


def nltk_stopwords_list():
    import nltk.corpus

    return [stopword for stopword in nltk.corpus.stopwords.words('english')]


def word_meronyms_part(word):
    """Get the part meronyms for the given word."""
    part_meronyms = []
    for synset in word_syn_sets(word):
        part_meronyms.append(synset.part_meronyms())
    return part_meronyms


def word_meronyms_member(word):
    """Get the member meronyms for the given word."""
    member_meronyms = []
    for synset in word_syn_sets(word):
        member_meronyms.append(synset.member_meronyms())
    return member_meronyms


def word_meronyms_substance(word):
    """Get the substance meronyms for the given word."""
    substance_meronyms = []
    for synset in word_syn_sets(word):
        substance_meronyms.append(synset.substance_meronyms())
    return substance_meronyms


def word_meronyms(word):
    """Find meronyms of the given word."""
    meronyms = {
        'part_meronyms': word_meronyms_part(word),
        'substance_meronyms': word_meronyms_substance(word),
        'member_meronyms': word_meronyms_member(word),
    }

    return meronyms


def word_holonyms_part(word):
    """Get the part holonyms for the given word."""
    part_holonyms = []
    for synset in word_syn_sets(word):
        part_holonyms.append(synset.part_holonyms())
    return part_holonyms


def word_holonyms_member(word):
    """Get the member holonyms for the given word."""
    member_holonyms = []
    for synset in word_syn_sets(word):
        member_holonyms.append(synset.member_holonyms())
    return member_holonyms


def word_holonyms_substance(word):
    """Get the substance holonyms for the given word."""
    substance_holonyms = []
    for synset in word_syn_sets(word):
        substance_holonyms.append(synset.substance_holonyms())
    return substance_holonyms


def word_holonyms(word):
    """Find holonyms of the given word."""
    holonyms = {
        'part_holonyms': word_holonyms_part(word),
        'substance_holonyms': word_holonyms_substance(word),
        'member_holonyms': word_holonyms_member(word),
    }

    return holonyms


def word_entailments(word):
    """Find other words representing the entailments of the given word."""
    entailments = []
    for synset in word_syn_sets(word):
        entailments.append(synset.entailments())
    return entailments


# TODO: the argument given to the two functions below should be a str or synset objects... it would probably make sense to have a string to synset decorator
def words_lowest_common_hypernyms(word_a, word_b):
    """Find the lowest common hypernyms for the given words."""

    if isinstance(word_a, str):
        # if we are given a string, take the first syn set for that string
        synset_a = word_syn_sets(word_a)[0]
    else:
        synset_a = word_a

    if isinstance(word_b, str):
        synset_b = word_syn_sets(word_b)[0]
    else:
        synset_b = word_b

    return synset_a.lowest_common_hypernyms(synset_b)


def words_semantic_similarity(word_a, word_b):
    """Find the semantic similarity of two words by determining which path ."""
    if isinstance(word_a, str):
        # if we are given a string, take the first syn set for that string
        synset_a = word_syn_sets(word_a)[0]
    else:
        synset_a = word_a

    if isinstance(word_b, str):
        synset_b = word_syn_sets(word_b)[0]
    else:
        synset_b = word_b

    return synset_a.path_similarity(synset_b)
