import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import decorators


def wordCloud(text, output_file_path=None):
    """Create a word cloud based on the given text."""
    import matplotlib.pyplot as plt
    from _utility import metaOutput
    from wordcloud import STOPWORDS, WordCloud

    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(text)

    figure, axes = plt.subplots()

    axes.axis("off")
    axes.imshow(wordcloud)
    plt.show()

    # TODO: it would be nice to save the image of the word cloud
    if output_file_path:
        figure.savefig(output_file_path)
        metaOutput('Saved an image of this word-cloud to "{}"'.format(output_file_path))

    return plt


def wordStem(word, stemmer='porter'):
    """Return the stem of the given word."""
    import nltk
    from _utility import metaOutput
    from strings import lowercase, titlecase

    available_stemmers = ['porter', 'lancaster']
    if lowercase(stemmer) not in available_stemmers:
        metaOutput('! Invalid stemmer given: {}\nAvailable stemmers are: {}'.format(stemmer, available_stemmers))
        return

    stemmer_object = eval('nltk.{}Stemmer()'.format(titlecase(stemmer)))
    return stemmer_object.stem(word)


def wordsGenerator(letters_list, min_word_length=2, required_characters_list=None):
    """Generate all possible words from the given list of letters."""
    from iterables import iterableNotIn

    valid_word_list = nltkWordList()
    letters = frequencyDistribution(''.join(letters_list))

    # TODO: I updated this function to use frequencyDistribution rather than nltk.freqDist... make sure this is working properly
    valid_words = [
        word for word in valid_word_list if len(word) >= min_word_length and frequencyDistribution(word) <= letters
    ]

    if required_characters_list is not None:
        # make sure the required characters are present
        # TODO: write a function for filtering
        valid_words = list(filter(lambda x: iterableNotIn(required_characters_list, x) == [], valid_words))

    return valid_words


def wordSearch(text, word):
    """Search for all instances of the given word in the given text."""
    t = nltkText(text)
    return t.concordance(word)


def wordSynSets(word):
    """Return the synonym sets for the given word."""
    from nltk.corpus import wordnet as wn

    return wn.synsets(word)


def wordSynonymsCommon(word):
    """Return the synonyms for the most common meaning of the given word."""
    return wordSynSets(word)[0].lemma_names()


def wordSynonyms(word):
    """Return the synonyms for all meanings of the given word."""
    synonyms = []
    for synset in wordSynSets(word):
        synonyms.append(synset.lemma_names())
    return synonyms


def wordSimilarWords(word):
    """Find words that are similar to the given word."""
    # TODO: implement!
    raise NotImplementedError


def wordDefinitionCommon(word):
    """Return the possible definitions of the given word."""
    return wordSynSets(word)[0].definition()


def wordUseInSentence(word):
    return wordSynSets(word)[0].examples()


def textValidEnglishWords(text: str):
    """Return the number of valid, English words in the given text."""
    from words_module import wordIsValidEnglishWord

    words_in_text = words(text)
    valid_english_words = []

    for word in words_in_text:
        if wordIsValidEnglishWord(word):
            valid_english_words.append(word)

    return valid_english_words


def textValidEnglishWordCount(text: str) -> int:
    """Return the number of valid, English words in the given text."""
    valid_english_words = textValidEnglishWords(text)
    valid_english_word_count = len(valid_english_words)
    return valid_english_word_count


def wordDefinitions(word):
    """Return the most common definition of the given word."""
    # TODO: this function could also be named "define"
    # TODO: also return the part of speach with each definition
    definitions = []
    for synset in wordSynSets(word):
        definitions.append(synset.definition())
    return definitions


def wordHyponymsCommon(word):
    """Return the hyponyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) of the most common meaning of the given word."""
    return [a.lemma_names() for a in wordSynSets(word)[0].hyponyms()]


def wordHyponyms(word):
    """Return the hyponyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) for all meanings of the given word."""
    hyponyms = []
    for synset in wordSynSets(word):
        hyponyms.append([a.lemma_names() for a in synset.hyponyms()])
    return hyponyms


def wordRootHypernym(word):
    """Return the root hypernym (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy)."""
    return [a.lemma_names() for a in wordSynSets(word)[0].root_hypernyms()]


def wordHypernymsCommon(word):
    """Return the hypernyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) of the most common meaning of the given word."""
    return [a.lemma_names() for a in wordSynSets(word)[0].hypernyms()]


def nltkText(text):
    """Return nltk.text.Text for the given text."""
    import nltk.text

    return nltk.text.Text(text)


def wordHypernyms(word):
    """Return the hypernyms (see https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) for all meanings of the given word."""
    hypernyms = []
    for synset in wordSynSets(word):
        hypernyms.append([a.lemma_names() for a in synset.hypernyms()])
    return hypernyms


def textTokens(text):
    """Return the tokens for the given text."""
    import nltk

    # TODO: this function could also be called "tokenize"
    return nltk.word_tokenize(text)


def frequencyDistribution(text):
    # TODO: is there a better way to get a freqdist that doesn't involve import nltk.book?
    from nltk.book import FreqDist

    return FreqDist(text)


def wordDispersionPlot(text, word_list):
    t = nltkText(text)
    # TODO: write a function to do this
    # make sure that the word_list is actually a list
    if isinstance(word_list, str):
        word_list = list(word_list)
    t.dispersion_plot(word_list)


def wordFrequency(text):
    from iterables import count

    word_list = words(text)
    return count(word_list)


def similarWords(text, word):
    """Find words which are used in a similar context as the given word."""
    t = nltkText(text)
    t.similar(word)


# TODO: this can also be called "lexical richness" or "lexical_diversity" (see http://www.nltk.org/book/ch01.html) - we should capture that somewhere
def wordRepitition(text):
    """Return the ratio of the number of unique words with the total number of words in the text."""
    from _utility import metaOutput

    unique_word_count = len(wordsUnique(text)) - 1
    word_count = len(words(text))
    # todo: the construct below is very similar to the stuff in the `sentenceAverageLength` function... could consolidate
    if word_count > 0:
        return 1 - (unique_word_count / word_count)
    else:
        metaOutput('No words found in the given text')
        return None


def wordCommonContexts(text, words_list):
    t = nltkText(text)
    return t.common_contexts(words_list)


def textPlotFrequency(text, limit=50, cumulative=False):
    # TODO: it would be cool to be able to print each of the most common words as a percent of the total words
    frequencyDistribution(text).plot(limit, cumulative=cumulative)


def textTabulateFrequency(text, limit=50, cumulative=False):
    """."""
    return frequencyDistribution(text).tabulate(limit, cumulative=cumulative)


def textHapaxes(text):
    """Return the hapaxes (the words that only occur once) in the given text."""
    hapaxes = frequencyDistribution(text).hapaxes()
    return hapaxes


def textCollocations(text):
    t = nltkText(text)
    # this function is weird in that it prints out the collocations, but doesn't return them
    t.collocations()


def wordRepititionPercent(text):
    """Return the percentage of the words which are repeated in the text."""
    from maths import percent

    word_repitition_ratio = wordRepitition(text)
    return percent(word_repitition_ratio)


def textTags(text):
    """Return each word in the text tagged with its part of speech."""
    # there is a helpful guide to the tags used by nltk here: https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
    blob = textBlob(text)
    return blob.tags


# TODO: write a function to get the part of speech from a word


def _textTagFilter(tags, tag_filter):
    """Filter the tags and return words whose tags match the given tag_filter."""
    return [tag for tag in tags if tag[1].startswith(tag_filter)]


def _textTagDeduplication(tags):
    """Deduplicate the tags and return only the name and not the part of speech."""
    from iterables import deduplicate

    return deduplicate([tag[0] for tag in tags])


def textNouns(text):
    """Get all nouns from the text."""
    tags = textTags(text)
    return _textTagDeduplication(_textTagFilter(tags, 'NN'))


def textVerbs(text):
    """Get all nouns from the text."""
    tags = textTags(text)
    return _textTagDeduplication(_textTagFilter(tags, 'VB'))


def properNouns(text):
    """Get all of the proper nouns in text."""
    tags = textTags(text)
    return _textTagDeduplication(_textTagFilter(tags, 'NNP'))


def properNounsCount(text):
    """."""
    from iterables import count

    tags = textTags(text)
    proper_nouns = [tag[0] for tag in _textTagFilter(tags, 'NNP')]
    return count(proper_nouns)


def wordsCount(text):
    """Return the number of words in the given text."""
    return len(words(text))


def wordCount(text, word, ignore_case=True):
    """Find the count of the given word in the text."""
    from strings import lowercase

    text_words = words(text)

    if ignore_case:
        text_words = lowercase(text_words)
        word = lowercase(word)

    return text_words.count(word)


def textContains(text, word, ignore_case=True):
    """Return whether or not the text contains the given word."""
    from strings import lowercase

    if ignore_case:
        text = lowercase(text)
        word = lowercase(word)

    return word in text


def tfidf(word, text, multiple_texts):
    """Find the "Term Frequency, Inverse Document Frequency" for the given word in the given text using the multiple texts."""
    import math

    tf = wordCount(text, word) / wordsCount(text)
    idf = math.log(len(multiple_texts) / (1 + sum([1 for t in multiple_texts if textContains(t, word)])))

    return tf * idf


def textSkeleton(text):
    """Return the verbs and nouns in the text."""
    from iterables import iterableCombine

    verbs = textVerbs(text)
    nouns = textNouns(text)
    return iterableCombine(nouns, verbs)


def wordsUnique(text):
    """Get a deduplicated list of all of the words in the given text."""
    from iterables import deduplicate
    from strings import lowercase

    word_list = [lowercase(word) for word in words(text)]
    unique_words = deduplicate(word_list)
    return unique_words


def subjectivity(string):
    ment = sentiment(string)
    return ment.subjectivity


def subjectivityNumberLine(string):
    from maths import numberLine

    return numberLine(subjectivity(string), 0, 1, 0.1)


def nltkWordList():
    """Return the nltk wordlist."""
    from nltk.corpus import words

    return words.words()


# TODO: this function should be able to remove stopwords from both a list and a string
def stopwordsRemove(string):
    from iterables import iterableNotIn
    from strings import lowercase
    from wordcloud import STOPWORDS

    # not sure if lowercasing this is the correct move, but if this is not done, words like "And" and "AND" will not be removed
    word_list = words(lowercase(string))
    # use nltk's stopwords
    non_stop_words = iterableNotIn(word_list, nltkStopwordsList())
    # use wordcloud's stopwords
    non_stop_words = iterableNotIn(non_stop_words, STOPWORDS)
    return ' '.join(non_stop_words)


def sentenceAverageLength(string):
    """Return the average length of a sentence in the string."""
    from _utility import metaOutput

    word_count = len(words(string))
    sentence_count = len(sentences(string))
    if sentence_count > 0:
        return word_count / sentence_count
    else:
        metaOutput('No sentences found in the given string')
        return None


def textBlob(string):
    """Return a textblob for the given string."""
    from textblob import TextBlob

    return TextBlob(string)


def sentences(string):
    blob = textBlob(string)
    return blob.sentences


def ngrams(string, n=3):
    blob = textBlob(string)
    # join the ngrams into strings
    grams = [' '.join(gram) for gram in blob.ngrams(n)]
    return grams


def ngramsCommon(string, n=3):
    from iterables import count

    grams = ngrams(string, n)
    if grams:
        sorted_grams = count(grams)
    else:
        sorted_grams = {}
    return sorted_grams


def nounPhrases(string):
    blob = textBlob(string)
    phrases = [phrase for phrase in blob.noun_phrases]
    return phrases


def nounPhrasesCommon(string):
    from iterables import count

    phrases = nounPhrases(string)
    return count(phrases)


def polarity(string):
    ment = sentiment(string)
    return ment.polarity


def polarityNumberLine(string):
    from maths import numberLine

    return numberLine(polarity(string), -1, 1, 0.1)


def words(text):
    blob = textBlob(text)
    return blob.words


def correctSpelling(string):
    blob = textBlob(string)
    return blob.correct()


def sentiment(string):
    blob = textBlob(string)
    return blob.sentiment


def nltkStopwordsList():
    import nltk.corpus

    return [stopword for stopword in nltk.corpus.stopwords.words('english')]


def wordMeronymsPart(word):
    """Get the part meronyms for the given word."""
    part_meronyms = []
    for synset in wordSynSets(word):
        part_meronyms.append(synset.part_meronyms())
    return part_meronyms


def wordMeronymsMember(word):
    """Get the member meronyms for the given word."""
    member_meronyms = []
    for synset in wordSynSets(word):
        member_meronyms.append(synset.member_meronyms())
    return member_meronyms


def wordMeronymsSubstance(word):
    """Get the substance meronyms for the given word."""
    substance_meronyms = []
    for synset in wordSynSets(word):
        substance_meronyms.append(synset.substance_meronyms())
    return substance_meronyms


def wordMeronyms(word):
    """Find meronyms of the given word."""
    meronyms = {
        'part_meronyms': wordMeronymsPart(word),
        'substance_meronyms': wordMeronymsSubstance(word),
        'member_meronyms': wordMeronymsMember(word),
    }

    return meronyms


def wordHolonymsPart(word):
    """Get the part holonyms for the given word."""
    part_holonyms = []
    for synset in wordSynSets(word):
        part_holonyms.append(synset.part_holonyms())
    return part_holonyms


def wordHolonymsMember(word):
    """Get the member holonyms for the given word."""
    member_holonyms = []
    for synset in wordSynSets(word):
        member_holonyms.append(synset.member_holonyms())
    return member_holonyms


def wordHolonymsSubstance(word):
    """Get the substance holonyms for the given word."""
    substance_holonyms = []
    for synset in wordSynSets(word):
        substance_holonyms.append(synset.substance_holonyms())
    return substance_holonyms


def wordHolonyms(word):
    """Find holonyms of the given word."""
    holonyms = {
        'part_holonyms': wordHolonymsPart(word),
        'substance_holonyms': wordHolonymsSubstance(word),
        'member_holonyms': wordHolonymsMember(word),
    }

    return holonyms


def wordEntailments(word):
    """Find other words representing the entailments of the given word."""
    entailments = []
    for synset in wordSynSets(word):
        entailments.append(synset.entailments())
    return entailments


# TODO: the argument given to the two functions below should be a str or synset objects... it would probably make sense to have a string to synset decorator
def wordsLowestCommonHypernyms(word_a, word_b):
    """Find the lowest common hypernyms for the given words."""

    if isinstance(word_a, str):
        # if we are given a string, take the first syn set for that string
        synset_a = wordSynSets(word_a)[0]
    else:
        synset_a = word_a

    if isinstance(word_b, str):
        synset_b = wordSynSets(word_b)[0]
    else:
        synset_b = word_b

    return synset_a.lowest_common_hypernyms(synset_b)


def wordsSemanticSimilarity(word_a, word_b):
    """Find the semantic similarity of two words by determining which path ."""
    if isinstance(word_a, str):
        # if we are given a string, take the first syn set for that string
        synset_a = wordSynSets(word_a)[0]
    else:
        synset_a = word_a

    if isinstance(word_b, str):
        synset_b = wordSynSets(word_b)[0]
    else:
        synset_b = word_b

    return synset_a.path_similarity(synset_b)
