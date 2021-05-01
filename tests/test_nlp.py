from d8s_nlp import (
    ngrams,
    ngramsCommon,
    tfidf,
    wordEntailments,
    wordHolonyms,
    wordHyponyms,
    wordHyponymsCommon,
    wordMeronyms,
    wordsGenerator,
    wordsLowestCommonHypernyms,
    wordsSemanticSimilarity,
    wordStem,
)


def test_ngramsCommon_1():
    assert ngramsCommon('this is just a foo foo') == {
        'this is just': 1,
        'is just a': 1,
        'just a foo': 1,
        'a foo foo': 1,
    }
    assert ngramsCommon('foo foo foo') == {'foo foo foo': 1}
    # handle situations where n is larger than the words in the string
    assert ngramsCommon('foo foo foo', n=4) == {}


def test_ngrams_1():
    n_grams = ngrams('this is just a test')
    assert n_grams == ['this is just', 'is just a', 'just a test']

    n_grams = ngrams('foo foo foo')
    assert n_grams == ['foo foo foo']

    n_grams = ngrams('foo foo foo', n=4)
    assert n_grams == []


def test_wordsGenerator():
    assert wordsGenerator(['a', 't', 'i']) == ['ai', 'ait', 'at', 'it', 'ta', 'tai', 'ti', 'at']
    # this example is based on http://www.nltk.org/book/ch02.html#fig-target
    assert wordsGenerator(tuple('egivrvonl'), min_word_length=4, required_characters_list=['r']) == (
        'enrol',
        'ergon',
        'genro',
        'girl',
        'girn',
        'giro',
        'giver',
        'glor',
        'glore',
        'glover',
        'goer',
        'goner',
        'gore',
        'gorlin',
        'govern',
        'grein',
        'grin',
        'groin',
        'grove',
        'grovel',
        'ignore',
        'inro',
        'involver',
        'iron',
        'irone',
        'levir',
        'lienor',
        'lier',
        'liner',
        'linger',
        'lire',
        'liver',
        'livor',
        'livre',
        'loir',
        'longer',
        'lore',
        'lori',
        'lorn',
        'lover',
        'lovering',
        'negro',
        'nigre',
        'noiler',
        'noir',
        'nori',
        'norie',
        'ogler',
        'ogre',
        'oiler',
        'oner',
        'oriel',
        'orle',
        'over',
        'overling',
        'regin',
        'region',
        'reign',
        'rein',
        'renvoi',
        'reoil',
        'revolving',
        'rigol',
        'rile',
        'rine',
        'ring',
        'ringe',
        'ringle',
        'rive',
        'rivel',
        'riven',
        'roil',
        'role',
        'rone',
        'rove',
        'roving',
        'vergi',
        'veri',
        'vier',
        'vigor',
        'viner',
        'violer',
        'vire',
        'vireo',
        'virl',
        'virole',
        'viron',
        'viver',
        'girl',
        'iron',
        'over',
        'ring',
    )


def test_wordHyponymsCommon():
    assert wordHyponymsCommon('sport') == [
        ['archery'],
        ['athletic_game'],
        ['blood_sport'],
        ['contact_sport'],
        ['cycling'],
        ['funambulism', 'tightrope_walking'],
        ['gymnastics', 'gymnastic_exercise'],
        ['judo'],
        ['outdoor_sport', 'field_sport'],
        ['racing'],
        ['riding', 'horseback_riding', 'equitation'],
        ['rock_climbing'],
        ['rowing', 'row'],
        ['skating'],
        ['skiing'],
        ['sledding'],
        ['spectator_sport'],
        ['team_sport'],
        ['track_and_field'],
        ['water_sport', 'aquatics'],
    ]


def test_wordHyponyms():
    assert wordHyponyms('sport') == [
        [
            ['archery'],
            ['athletic_game'],
            ['blood_sport'],
            ['contact_sport'],
            ['cycling'],
            ['funambulism', 'tightrope_walking'],
            ['gymnastics', 'gymnastic_exercise'],
            ['judo'],
            ['outdoor_sport', 'field_sport'],
            ['racing'],
            ['riding', 'horseback_riding', 'equitation'],
            ['rock_climbing'],
            ['rowing', 'row'],
            ['skating'],
            ['skiing'],
            ['sledding'],
            ['spectator_sport'],
            ['team_sport'],
            ['track_and_field'],
            ['water_sport', 'aquatics'],
        ],
        [
            ['professional_baseball'],
            ['professional_basketball'],
            ['professional_boxing'],
            ['professional_football'],
            ['professional_golf'],
            ['professional_tennis'],
            ['professional_wrestling'],
            ['sumo'],
        ],
        [],
        [],
        [],
        [['freak', 'monster', 'monstrosity', 'lusus_naturae']],
        [
            ['drollery', 'clowning', 'comedy', 'funniness'],
            ['jocosity', 'jocularity'],
            ['pun', 'punning', 'wordplay', 'paronomasia'],
            ['waggery', 'waggishness'],
        ],
        [],
        [],
    ]


def test_porterStemmer():
    assert wordStem('lying') == 'lie'
    assert wordStem('lying', stemmer='porter') == 'lie'


def test_lancasterStemmer():
    assert wordStem('lying', stemmer='lancaster') == 'lying'
    assert wordStem('lying', stemmer='Lancaster') == 'lying'
    assert wordStem('lying', stemmer='LANCASTER') == 'lying'


def test_tfidf():
    s1 = """Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known."""

    s2 = """Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films."""

    s3 = """The Colt Python is a .357 Magnum caliber revolver formerly
 manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
 It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
 in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
 Colt Python targeted the premium revolver market segment. Some firearm
 collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
 Thompson, Renee Smeets and Martin Dougherty have described the Python as the
 finest production revolver ever made."""

    assert tfidf('python', s1, [s1, s2, s3]) == 0.0
    assert round(tfidf('Magnum', s3, [s1, s2, s3]), 6) == 0.013667


def test_wordMeronyms_1():
    assert (
        str(wordMeronyms("tree"))
        == "{'part_meronyms': [[Synset('burl.n.02'), Synset('crown.n.07'), Synset('limb.n.02'), Synset('stump.n.01'), Synset('trunk.n.01')], [], [], [], [], [], []], 'substance_meronyms': [[Synset('heartwood.n.01'), Synset('sapwood.n.01')], [], [], [], [], [], []], 'member_meronyms': [[], [], [], [], [], [], []]}"
    )


def test_wordHolonyms_1():
    assert (
        str(wordHolonyms("tree"))
        == "{'part_holonyms': [[], [], [], [], [], [], []], 'substance_holonyms': [[], [], [], [], [], [], []], 'member_holonyms': [[Synset('forest.n.01')], [], [], [], [], [], []]}"
    )


def test_wordEntailments_1():
    assert str(wordEntailments("eat")) == "[[Synset('chew.v.01'), Synset('swallow.v.01')], [], [], [], [], []]"


def test_wordsLowestCommonHypernyms_1():
    assert str(wordsLowestCommonHypernyms('right_whale', 'minke_whale')) == "[Synset('baleen_whale.n.01')]"


def test_wordsSemanticSimilarity_1():
    assert wordsSemanticSimilarity('right_whale', 'minke_whale') == 0.25
