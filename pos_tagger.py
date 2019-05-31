import pymorphy2 as morph
from nltk.corpus import stopwords
from pymystem3 import Mystem
m = Mystem()
morph = morph.MorphAnalyzer()
 
def strip_punctuation(s):
    #I tried to add all the possible pinctuation but the dataset still requires some manual fixing
    punct = '!\"#$%&\'()*+,./:;<=>?@[\]^`{|}~«»“”–—'
    return ''.join(c for c in s if c not in punct)
 
def lemmatize(text):
    text = strip_punctuation(text)
    lemmas = m.lemmatize(text)
    #filtering stopwords and words with less than 3 symbols
    lemmas[:] = [lemma for lemma in lemmas if lemma not in stopwords.words('russian') and len(lemma.strip()) > 2]
    return(lemmas)
 
def pos_tagging(lemmas):
    #filtering words with meaning
    tags = ['NOUN', 'VERB', 'ADVB', 'ADJF', 'INFN']
    #filtering types of proper nouns
    props = ['Name', 'Patr', 'Geox', 'Abbr']
    line_tags = [] 
    for lemma in lemmas:
        #removing all the words with digits
        if not any(char.isdigit() for char in lemma):
            try:
                #tagging proper nouns
                if any(prop in str(morph.parse(lemma)[0].tag) for prop in props):
                    lemma_tagged = lemma + '_PROPN'
                elif morph.parse(lemma)[0].tag.POS in tags:
                    lemma_tagged = lemma + '_' + morph.parse(lemma)[0].tag.POS
                line_tags.append(lemma_tagged)
            except:
                pass
    return(line_tags)

def universal_tags(lemmas):
    #we need to replace pymorphy2 tags with Universal tags so it would work with tagged vector models like ruscorpora
    universal = []
    for lemma in lemmas:
        if lemma.endswith('INFN'):
            universal.append(lemma.replace('INFN', 'VERB'))
        elif lemma.endswith('ADJF'):
            universal.append(lemma.replace('ADJF', 'ADJ'))
        elif lemma.endswith('ADVB'):
            universal.append(lemma.replace('ADVB', 'ADV'))
        else:
            universal.append(lemma)
    return(' '.join(universal))
    
def POS_main(text):
    lemmas = lemmatize(text)
    lemmas = pos_tagging(lemmas)
    lemmas = universal_tags(lemmas)
    return(lemmas)