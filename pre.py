import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize  # Tokenisasi teks
from nltk.corpus import stopwords

slangwords = {
    "gr8": "great",
    "lol": "laugh out loud",
    "omg": "oh my god",
    "btw": "by the way",
    "idk": "I don't know",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "thx": "thanks",
    "pls": "please",
    "jk": "just kidding",
    "np": "no problem",
    "wyd": "what you doing",
    "irl": "in real life",
    "afaik": "as far as I know",
    "ama": "ask me anything",
    "dm": "direct message",
    "ftw": "for the win",
    "gtg": "got to go",
    "hmu": "hit me up",
    "icymi": "in case you missed it",
    "nvm": "never mind",
    "ofc": "of course",
    "rofl": "rolling on floor laughing",
    "tmi": "too much information",
    "yolo": "you only live once",
    "bff": "best friends forever",
    "fomo": "fear of missing out",
    "ily": "I love you",
    "lmk": "let me know",
    "nsfw": "not safe for work",
    "ootd": "outfit of the day",
    "ttyl": "talk to you later",
    "wtf": "what the fuck",
    "xoxo": "hugs and kisses",
    "bae": "before anyone else",
    "diy": "do it yourself",
    "fyi": "for your information",
    "hbd": "happy birthday",
    "idc": "I don't care",
    "kys": "kill yourself",
    "omw": "on my way",
    "pm": "private message",
    "rip": "rest in peace",
    "smd": "suck my dick",
    "tbt": "throwback Thursday",
    "wth": "what the hell",
    "yw": "you're welcome",
    "zzz": "sleeping or bored"
}

def fix_slangwords(text):
    words = text.split()
    fixed_words = []

    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = re.sub(r'[^\w\s]', '', text) # remove numbers
    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text




def filteringText(tokens):  # Terima list of tokens
    stop_words = set(stopwords.words('english'))
    extra_stopwords = {'yeah', 'uh', 'hmm', 'umm', 'like', 'okay', 'ok', 'ya'}
    stop_words.update(extra_stopwords)

    # Filter kata yang bukan stopword dan bukan tanda baca
    filtered = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]

    return filtered

# Konversi POS tag ke WordNet format
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

# Fungsi lemmatization
def lemmatizationText(tokens):  # menerima list token, bukan string
    lemmatizer = WordNetLemmatizer()

    # Tagging POS dari token
    pos_tags = pos_tag(tokens)

    # Lemmatization berdasarkan POS
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

    return lemmatized  # atau ' '.join(lemmatized) kalau ingin jadi kalimat


def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence