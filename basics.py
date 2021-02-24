import os
import numpy as np
from numpy import linalg as LA
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
import gensim.downloader as api
gensim_model = api.load(name='word2vec-google-news-300')
gensim_model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.


def assure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def word_movers_distance(sequence_x, sequence_y):
    return gensim_model.wmdistance(sequence_x, sequence_y)


def l2_distance(sequence_x, sequence_y):
    return LA.norm(x=np.subtract(sequence_x, sequence_y).ravel(), ord=2)
    # return math.sqrt(np.sum(np.square(np.subtract(sequence_x, sequence_y))))


def l1_distance(sequence_x, sequence_y):
    return LA.norm(x=np.subtract(sequence_x, sequence_y).ravel(), ord=1)
    # return np.sum(np.absolute(np.subtract(sequence_x, sequence_y)))


def linf_distance(sequence_x, sequence_y):
    return LA.norm(x=np.subtract(sequence_x, sequence_y).ravel(), ord=np.inf)


def l0_distance(sequence_x, sequence_y):
    return LA.norm(x=np.subtract(sequence_x, sequence_y).ravel(), ord=0)
    # return np.count_nonzero(np.absolute(np.subtract(sequence_x, sequence_y)))


tag_map = {
    'CC': None,  # coordin. conjunction (and, but, or)
    'CD': wn.NOUN,  # cardinal number (one, two)
    'DT': None,  # determiner (a, the)
    'EX': wn.ADV,  # existential ‘there’ (there)
    'FW': None,  # foreign word (mea culpa)
    'IN': wn.ADV,  # preposition/sub-conj (of, in, by)
    'JJ': [wn.ADJ, wn.ADJ_SAT],  # adjective (yellow)
    'JJR': [wn.ADJ, wn.ADJ_SAT],  # adj., comparative (bigger)
    'JJS': [wn.ADJ, wn.ADJ_SAT],  # adj., superlative (wildest)
    'LS': None,  # list item marker (1, 2, One)
    'MD': None,  # modal (can, should)
    'NN': wn.NOUN,  # noun, sing. or mass (llama)
    'NNS': wn.NOUN,  # noun, plural (llamas)
    'NNP': wn.NOUN,  # proper noun, sing. (IBM)
    'NNPS': wn.NOUN,  # proper noun, plural (Carolinas)
    'PDT': [wn.ADJ, wn.ADJ_SAT],  # predeterminer (all, both)
    'POS': None,  # possessive ending (’s )
    'PRP': None,  # personal pronoun (I, you, he)
    'PRP$': None,  # possessive pronoun (your, one’s)
    'RB': wn.ADV,  # adverb (quickly, never)
    'RBR': wn.ADV,  # adverb, comparative (faster)
    'RBS': wn.ADV,  # adverb, superlative (fastest)
    'RP': [wn.ADJ, wn.ADJ_SAT],  # particle (up, off)
    'SYM': None,  # symbol (+,%, &)
    'TO': None,  # “to” (to)
    'UH': None,  # interjection (ah, oops)
    'VB': wn.VERB,  # verb base form (eat)
    'VBD': wn.VERB,  # verb past tense (ate)
    'VBG': wn.VERB,  # verb gerund (eating)
    'VBN': wn.VERB,  # verb past participle (eaten)
    'VBP': wn.VERB,  # verb non-3sg pres (eat)
    'VBZ': wn.VERB,  # verb 3sg pres (eats)
    'WDT': None,  # wh-determiner (which, that)
    'WP': None,  # wh-pronoun (what, who)
    'WP$': None,  # possessive (wh- whose)
    'WRB': None,  # wh-adverb (how, where)
    '$': None,  # dollar sign ($)
    '#': None,  # pound sign (#)
    '“': None,  # left quote (‘ or “)
    '”': None,  # right quote (’ or ”)
    '(': None,  # left parenthesis ([, (, {, <)
    ')': None,  # right parenthesis (], ), }, >)
    ',': None,  # comma (,)
    '.': None,  # sentence-final punc (. ! ?)
    ':': None  # mid-sentence punc (: ; ... – -)
}