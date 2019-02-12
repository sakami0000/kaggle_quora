import re
import string
import unicodedata

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# -------------
# preprocess 1
# -------------

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&',
          '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
          '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
          '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
          '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕',
          '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦',
          '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│',
          '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬',
          '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}


def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_misspell(misspell_dict):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re


def replace_typical_misspell(text):
    misspellings, misspellings_re = _get_misspell(misspell_dict)

    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)


sia = SIA()


def add_features(embeddings_index, df):
    df['question_text'] = df['question_text'].map(str)

    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(
        lambda row: float(row['capitals']) / float(row['total_length']), axis=1)

    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(
        lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    df['oov'] = df['question_text'].apply(
        lambda comment: sum(1 for w in comment.split() if embeddings_index.get(w) is None))
    df['oov_vs_words'] = df['oov'] / df['num_words']

    df['compound'] = df['question_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    return df


# -------------
# preprocess 2
# -------------


def clean_misspell(text):
    """
    misspell list (quora vs. glove)
    """
    misspell_to_sub = {
        'Terroristan': 'terrorist Pakistan',
        'terroristan': 'terrorist Pakistan',
        'BIMARU': 'Bihar, Madhya Pradesh, Rajasthan, Uttar Pradesh',
        'Hinduphobic': 'Hindu phobic',
        'hinduphobic': 'Hindu phobic',
        'Hinduphobia': 'Hindu phobic',
        'hinduphobia': 'Hindu phobic',
        'Babchenko': 'Arkady Arkadyevich Babchenko faked death',
        'Boshniaks': 'Bosniaks',
        'Dravidanadu': 'Dravida Nadu',
        'mysoginists': 'misogynists',
        'MGTOWS': 'Men Going Their Own Way',
        'mongloid': 'Mongoloid',
        'unsincere': 'insincere',
        'meninism': 'male feminism',
        'jewplicate': 'jewish replicate',
        'unoin': 'Union',
        'daesh': 'Islamic State of Iraq and the Levant',
        'Kalergi': 'Coudenhove-Kalergi',
        'Bhakts': 'Bhakt',
        'bhakts': 'Bhakt',
        'Tambrahms': 'Tamil Brahmin',
        'Pahul': 'Amrit Sanskar',
        'SJW': 'social justice warrior',
        'SJWs': 'social justice warrior',
        'incel': ' involuntary celibates',
        'incels': ' involuntary celibates',
        'emiratis': 'Emiratis',
        'weatern': 'western',
        'westernise': 'westernize',
        'Pizzagate': 'Pizzagate conspiracy theory',
        'naïve': 'naive',
        'Skripal': 'Sergei Skripal',
        'Remainers': 'British remainer',
        'remainers': 'British remainer',
        'bremainer': 'British remainer',
        'antibrahmin': 'anti Brahminism',
        'HYPSM': 'Harvard, Yale, Princeton, Stanford, MIT',
        'HYPS': 'Harvard, Yale, Princeton, Stanford',
        'kompromat': 'compromising material',
        'Tharki': 'pervert',
        'tharki': 'pervert',
        'mastuburate': 'masturbate',
        'Zoë': 'Zoe',
        'indans': 'Indian',
        'xender': 'gender',
        'Naxali ': 'Naxalite ',
        'Naxalities': 'Naxalites',
        'Bathla': 'Namit Bathla',
        'Mewani': 'Indian politician Jignesh Mevani',
        'clichéd': 'cliche',
        'cliché': 'cliche',
        'clichés': 'cliche',
        'Wjy': 'Why',
        'Fadnavis': 'Indian politician Devendra Fadnavis',
        'Awadesh': 'Indian engineer Awdhesh Singh',
        'Awdhesh': 'Indian engineer Awdhesh Singh',
        'Khalistanis': 'Sikh separatist movement',
        'madheshi': 'Madheshi',
        'BNBR': 'Be Nice, Be Respectful',
        'Bolsonaro': 'Jair Bolsonaro',
        'XXXTentacion': 'Tentacion',
        'Padmavat': 'Indian Movie Padmaavat',
        'Žižek': 'Slovenian philosopher Slavoj Žižek',
        'Adityanath': 'Indian monk Yogi Adityanath',
        'Brexit': 'British Exit',
        'Brexiter': 'British Exit supporter',
        'Brexiters': 'British Exit supporters',
        'Brexiteer': 'British Exit supporter',
        'Brexiteers': 'British Exit supporters',
        'Brexiting': 'British Exit',
        'Brexitosis': 'British Exit disorder',
        'brexit': 'British Exit',
        'brexiters': 'British Exit supporters',
        'jallikattu': 'Jallikattu',
        'fortnite': 'Fortnite ',
        'Swachh': 'Swachh Bharat mission campaign ',
        'Quorans': 'Quoran',
        'Qoura ': 'Quora ',
        'quoras': 'Quora',
        'Quroa': 'Quora',
        'QUORA': 'Quora',
        'narcissit': 'narcissist',

        # extra in sample
        'Doklam': 'Tibet',
        'Drumpf': 'Donald Trump fool',
        'Drumpfs': 'Donald Trump fools',
        'Strzok': 'Hillary Clinton scandal',
        'rohingya': 'Rohingya ',
        'wumao': 'cheap Chinese stuff',
        'wumaos': 'cheap Chinese stuff',
        'Sanghis': 'Sanghi',
        'Tamilans': 'Tamils',
        'biharis': 'Biharis',
        'Rejuvalex': 'hair growth formula',
        'Feku': 'Fake',
        'deplorables': 'deplorable',
        'muhajirs': 'Muslim immigrant',
        'Gujratis': 'Gujarati',
        'Chutiya': 'Fucker',
        'Chutiyas': 'Fucker',
        'thighing': 'masturbate',
        '卐': 'Nazi Germany',
        'Pribumi': 'Native Indonesian',
        'Gurmehar': 'Gurmehar Kaur Indian student activist',
        'Novichok': 'Soviet Union agents',
        'Khazari': 'Khazars',
        'Demonetization': 'demonetization',
        'demonitisation': 'demonetization',
        'demonitization': 'demonetization',
        'cryptocurrencies': 'cryptocurrency',
        'Hindians': 'North Indian who hate British',
        'vaxxer': 'vocal nationalist ',
        'remoaner': 'remainer ',
        'bremoaner': 'British remainer ',
        'Jewism': 'Judaism',
        'Eroupian': 'European',
        'WMAF': 'White male married Asian female',
        'moeslim': 'Muslim',
        'cishet': 'cisgender and heterosexual person',
        'Eurocentric': 'Eurocentrism ',
        'Jewdar': 'Jew dar',
        'Asifa': 'abduction, rape, murder case ',
        'marathis': 'Marathi',
        'Trumpanzees': 'Trump chimpanzee fool',
        'Crimean': 'Crimea people ',
        'atrracted': 'attract',
        'LGBT': 'lesbian, gay, bisexual, transgender',
        'Boshniak': 'Bosniaks ',
        'Myeshia': 'widow of Green Beret killed in Niger',
        'demcoratic': 'Democratic',
        'raaping': 'rape',
        'Dönmeh': 'Islam',
        'feminazism': 'feminism nazi',
        'langague': 'language',
        'Hongkongese': 'HongKong people',
        'hongkongese': 'HongKong people',
        'Kashmirians': 'Kashmirian',
        'Chodu': 'fucker',
        'penish': 'penis',
        'micropenis': 'tiny penis',
        'Madridiots': 'Real Madrid idiot supporters',
        'Ambedkarite': 'Dalit Buddhist movement ',
        'ReleaseTheMemo': 'cry for the right and Trump supporters',
        'harrase': 'harass',
        'Barracoon': 'Black slave',
        'Castrater': 'castration',
        'castrater': 'castration',
        'Rapistan': 'Pakistan rapist',
        'rapistan': 'Pakistan rapist',
        'Turkified': 'Turkification',
        'turkified': 'Turkification',
        'Dumbassistan': 'dumb ass Pakistan',
        'facetards': 'Facebook retards',
        'rapefugees': 'rapist refugee',
        'superficious': 'superficial',

        # extra from kagglers
        'colour': 'color',
        'centre': 'center',
        'favourite': 'favorite',
        'travelling': 'traveling',
        'counselling': 'counseling',
        'theatre': 'theater',
        'cancelled': 'canceled',
        'labour': 'labor',
        'organisation': 'organization',
        'wwii': 'world war 2',
        'citicise': 'criticize',
        'youtu ': 'youtube ',
        'sallary': 'salary',
        'Whta': 'What',
        'narcisist': 'narcissist',
        'howdo': 'how do',
        'whatare': 'what are',
        'howcan': 'how can',
        'howmuch': 'how much',
        'howmany': 'how many',
        'whydo': 'why do',
        'doI': 'do I',
        'theBest': 'the best',
        'howdoes': 'how does',
        'mastrubation': 'masturbation',
        'mastrubate': 'masturbate',
        'mastrubating': 'masturbating',
        'pennis': 'penis',
        'Etherium': 'Ethereum',
        'bigdata': 'big data',
        '2k17': '2017',
        '2k18': '2018',
        'qouta': 'quota',
        'exboyfriend': 'ex boyfriend',
        'airhostess': 'air hostess',
        'whst': 'what',
        'watsapp': 'whatsapp',

        # extra
        'bodyshame': 'body shaming',
        'bodyshoppers': 'body shopping',
        'bodycams': 'body cams',
        'Cananybody': 'Can any body',
        'deadbody': 'dead body',
        'deaddict': 'de addict',
        'Northindian': 'North Indian ',
        'northindian': 'north Indian ',
        'northkorea': 'North Korea',
        'Whykorean': 'Why Korean',
        'koreaboo': 'Korea boo ',
        'Brexshit': 'British Exit bullshit',
        'shithole': 'shithole ',
        'shitpost': 'shit post',
        'shitslam': 'shit Islam',
        'shitlords': 'shit lords',
        'Fck': 'Fuck',
        'fck': 'fuck',
        'Clickbait': 'click bait ',
        'clickbait': 'click bait ',
        'mailbait': 'mail bait',
        'healhtcare': 'healthcare',
        'trollbots': 'troll bots',
        'trollled': 'trolled',
        'trollimg': 'trolling',
        'cybertrolling': 'cyber trolling',
        'sickular': 'India sick secular ',
        'suckimg': 'sucking',
        'Idiotism': 'idiotism',
        'Niggerism': 'Nigger',
        'Niggeriah': 'Nigger'
    }
    misspell_re = re.compile('(%s)' % '|'.join(misspell_to_sub.keys()))

    def _replace(match):
        try:
            word = misspell_to_sub.get(match.group(0))
        except KeyError:
            word = match.group(0)
            print('!!Error: Could Not Find Key: {}'.format(word))
        return word

    return misspell_re.sub(_replace, text)


def spacing_misspell(text):
    """
    'deadbody' -> 'dead body'
    """
    misspell_list = [
        '(F|f)uck',
        'Trump',
        '\W(A|a)nti',
        '(W|w)hy',
        '(W|w)hat',
        'How',
        'care\W',
        '\Wover',
        'gender',
        'people',
    ]
    misspell_re = re.compile('(%s)' % '|'.join(misspell_list))
    return misspell_re.sub(r' \1 ', text)


def clean_latex(text):
    """
    convert r"[math]\vec{x} + \vec{y}" to English
    """
    # edge case
    text = re.sub(r'\[math\]', ' LaTex math ', text)
    text = re.sub(r'\[\/math\]', ' LaTex math ', text)
    text = re.sub(r'\\', ' LaTex ', text)

    pattern_to_sub = {
        r'\\mathrm': ' LaTex math mode ',
        r'\\mathbb': ' LaTex math mode ',
        r'\\boxed': ' LaTex equation ',
        r'\\begin': ' LaTex equation ',
        r'\\end': ' LaTex equation ',
        r'\\left': ' LaTex equation ',
        r'\\right': ' LaTex equation ',
        r'\\(over|under)brace': ' LaTex equation ',
        r'\\text': ' LaTex equation ',
        r'\\vec': ' vector ',
        r'\\var': ' variable ',
        r'\\theta': ' theta ',
        r'\\mu': ' average ',
        r'\\min': ' minimum ',
        r'\\max': ' maximum ',
        r'\\sum': ' + ',
        r'\\times': ' * ',
        r'\\cdot': ' * ',
        r'\\hat': ' ^ ',
        r'\\frac': ' / ',
        r'\\div': ' / ',
        r'\\sin': ' Sine ',
        r'\\cos': ' Cosine ',
        r'\\tan': ' Tangent ',
        r'\\infty': ' infinity ',
        r'\\int': ' integer ',
        r'\\in': ' in ',
    }

    # post process for look up
    pattern_dict = {k.strip('\\'): v for k, v in pattern_to_sub.items()}

    # init re
    patterns = pattern_to_sub.keys()
    pattern_re = re.compile('(%s)' % '|'.join(patterns))

    def _replace(match):
        """
        reference: https://www.kaggle.com/hengzheng/attention-capsule-why-not-both-lb-0-694 # noqa
        """
        word = pattern_dict.get(match.group(0).strip('\\'))
        return word

    return pattern_re.sub(_replace, text)


def normalize_unicode(text):
    """
    unicode string normalization
    """
    return unicodedata.normalize('NFKD', text)


def remove_newline(text):
    """
    remove \n and  \t
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\b', ' ', text)
    text = re.sub('\r', ' ', text)
    return text


def decontracted(text):
    """
    de-contract the contraction
    """
    # specific
    text = re.sub(r'[Ww]on[\'’]t', 'will not', text)
    text = re.sub(r'[Cc]an[\'’]t', 'can not', text)
    text = re.sub(r'[Yy][\'’]all', 'you all', text)
    text = re.sub(r'[Yy]a[\'’]ll', 'you all', text)

    # general
    text = re.sub(r'[Ii][\'’]m', 'i am', text)
    text = re.sub(r'[Aa]in[\'’]t', 'is not', text)
    text = re.sub(r'n[\'’]t', ' not', text)
    text = re.sub(r'[\'’]re', ' are', text)
    text = re.sub(r'[\'’]s', ' is', text)
    text = re.sub(r'[\'’]d', ' would', text)
    text = re.sub(r'[\'’]ll', ' will', text)
    text = re.sub(r'[\'’]t', ' not', text)
    text = re.sub(r'[\'’]ve', ' have', text)
    return text


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    regular_punct = list(string.punctuation)
    extra_punct = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&',
        '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
        '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
        '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
        '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
        '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
        'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
        '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï',
        'Ø', '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
    puncts = regular_punct + extra_punct
    for punct in puncts:
        if punct in text:
            text = text.replace(punct, f' {punct} ')
    return text


def spacing_digit(text):
    """
    add space before and after digits
    """
    re_tok = re.compile('([0-9])')
    return re_tok.sub(r' \1 ', text)


def spacing_number(text):
    """
    add space before and after numbers
    """
    re_tok = re.compile('([0-9]+)')
    return re_tok.sub(r' \1 ', text)


def remove_number(text):
    """
    numbers are not toxic
    """
    return re.sub('\d+', ' ', text)


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+$', '', text)
    return text


def extract_features(text):
    """
    embeddings_index from global name space
    """
    num_words = len(text.split())
    oov = sum(1 for w in text.split()
              if embeddings_index.get(w) is None
              and embeddings_index.get(w.lower()) is None)
    oov_vs_words = oov / num_words

    return oov_vs_words


def preprocess(text, remove_num=True):
    """
    preprocess text into clean text for tokenization
    NOTE:
        1. glove supports uppper case words
        2. glove supports digit
        3. glove supports punctuation
        5. glove supports domains e.g. www.apple.com
        6. glove supports misspelled words e.g. FUCKKK
    """
    # 1. normalize
    text = normalize_unicode(text)
    # 2. remove new line
    text = remove_newline(text)
    # 3. de-contract
    text = decontracted(text)
    # 4. clean misspell
    text = clean_misspell(text)
    # 5. space misspell
    text = spacing_misspell(text)
    # 6. clean_latex
    text = clean_latex(text)
    # 7. space
    text = spacing_punctuation(text)
    # 8. handle number
    if remove_num:
        text = remove_number(text)
    else:
        text = spacing_digit(text)
    # 9. remove space
    text = remove_space(text)
    # 10. extract features
    oov_vs_words = extract_features(text)

    return [text, oov_vs_words]


# ---------------------
# Logistic Regression
# ---------------------


class NBFeaturer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)

    def fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb


def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()
