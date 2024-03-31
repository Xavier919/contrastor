import spacy
import re
import string
import html
nlp = spacy.load("fr_core_news_md")

def expand_contractions(text):
    update_text = []
    text = text.split(" ")
    for word in text:
        words = word.split("’")
        for word_ in words:
            update_text.append(word_)
    return (" ").join(update_text)

def spacy_tokenizer(text):
    nlp = spacy.load("fr_core_news_md")
    return [tok.text for tok in nlp.tokenizer(str(text))]

def text_edit(dataset,grp_num=False,rm_newline=False,rm_punctuation=False,lowercase=False,lemmatize=False,html_=False,expand=False):

    extended_punctuation = string.punctuation + "«»…“”–—-"
    pattern = re.compile(f"[{re.escape(extended_punctuation)}]")

    for attrs in dataset.values():
        text_ = attrs["text"]

        if lowercase:
            text_ = text_.lower()

        if expand:
            text_ = expand_contractions(text_)

        if html_:
            text_ = html.unescape(text_)
            text_ = re.sub('\xa0', ' ', text_)
            text_ = re.sub('\u2060', ' ', text_)

        if grp_num:
            text_ = re.sub(r"\d+", "num", text_)

        if rm_newline:
            text_ = re.sub(r"\n(\w)", r"\1", text_)

        if rm_punctuation:
            text_ = pattern.sub("", text_)
            text_ = re.sub(r" +", " ", text_)
            text_ = text_.replace("\u2060num", "")

        if lemmatize:
            text_words = text_.split()
            text_ = " ".join(tok.lemma_ for tok in nlp(" ".join(text_words)))

        attrs["text"] = text_

    return dataset
