"""
"""
from enum import Enum
from jisho_api.util import CLITagger
from jisho_api.cli import console
from rich.markdown import Markdown
from pathlib import Path
import pprint
import json
from typing import List
import copy
import time
import pandas as pd
import os
from os.path import join

from jisho_api.word import Word
from jisho_api.kanji import Kanji
from jisho_api.sentence import Sentence
from jisho_api.tokenize import Tokens
from jisho_api.cli import scrape

import unicodedata  # to detect if kanji or kana
import re
import numpy as np
import collections
import math

import urllib
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
import requests
URL = "https://jisho.org/search/"
from googletrans import Translator
# %%


class RequestMeta(BaseModel):
    status: int


class PosTag(Enum):
    adj = "Adjective"
    adv = "Adverb"
    conj = "Conjunction"
    det = "Determiner"
    interjection = "Interjection"
    noun = "Noun"
    particle = "Particle"
    pr_noun = "Proper noun"
    prfx = "Prefix"
    pron = "Pronoun"
    sfx = "Suffix"
    unk = "Unknown"
    verb = "Verb"

    # rather than causing the program to crash, inform the user of the unexpected posTag
    # implementation source: https://stackoverflow.com/questions/44867597/is-there-a-way-to-specify-a-default-value-for-python-enums
    @classmethod
    def _missing_(self, value):
        print("Unexpected positional Tag: {}".format(value))
        return self.unk


class TokenConfig(BaseModel):
    token: str
    pos_tag: PosTag
    #pos_tag: str


class TokenRequest(BaseModel):
    meta: RequestMeta
    data: List[TokenConfig]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        yield from self.data

    def rich_print(self):
        base = ''
        toks = ''
        for i, d in enumerate(self):
            base += CLITagger.underline(d.token) + ' '
            toks += f"{i}. {d.token} [violet][{str(d.pos_tag.value)}][/violet]\n"
        console.print(base)
        console.print(toks)


# %%
text_path = "text"

path_pla_common_hir = join(text_path, "PLA_text", "common", "ja-hiragana.txt")
path_pla_common_kat = join(text_path, "PLA_text", "common", "ja-katakana.txt")
path_pla_story_hir = join(text_path, "PLA_text", "story", "ja-hiragana.txt")
path_pla_story_kat = join(text_path, "PLA_text", "story", "ja-katakana.txt")

path_pla_common_en = join(text_path, "PLA_text", "common", "en.txt")
path_pla_story_en = join(text_path, "PLA_text", "story", "en.txt")


os.path.exists(path_pla_common_en)
os.path.exists(path_pla_story_en)


with open(path_pla_common_hir, encoding='utf16') as f:
    pla_common_hir = f.readlines()
with open(path_pla_common_kat, encoding='utf16') as f:
    pla_common_kat = f.readlines()
with open(path_pla_story_hir, encoding='utf16') as f:
    pla_story_hir = f.readlines()
with open(path_pla_story_kat, encoding='utf16') as f:
    pla_story_kat = f.readlines()
with open(path_pla_common_en, encoding='utf16') as f:
    pla_common_en = f.readlines()
with open(path_pla_story_en, encoding='utf16') as f:
    pla_story_en = f.readlines()


remove_characters = ["\n", "\t", "\ue30a", "\ue30b", "\ue30c", "\ue30d", "\ue30e", "\ue30f", "\ue301", "\ue302", "\ue303", "\ue304", "\ue305", "\ue306", "\ue307",
                     "\ue308", "\ue309", "\ue310", "\ue31a", "\ue31b", "\ue31c", "\ue31d", "\ue31e", "\ue311", "\ue312", "\ue313", "\ue314", "\ue315", "\ue316", "\ue317", "\ue319", "\u3000"]
# %% Get all kanji

kanji_list = []


for i in range(len(pla_common_hir)):
    line = pla_common_hir[i]

    for character in remove_characters:
        line = line.replace(character, "")

    print(f"\r{i}/{len(pla_common_hir)}; {np.round(i/len(pla_common_hir)*100,1)}   ", end = "\r")
    for l in range(len(line)):
        letter = line[l]
        unic = unicodedata.name(letter)
        # print(unic)
        if "CJK" in unic:
            kanji_list.append(letter)

for i in range(len(pla_story_hir)):
    line = pla_story_hir[i]

    for character in remove_characters:
        line = line.replace(character, "")

    print(f"\r{i}/{len(pla_story_hir)}; {np.round(i/len(pla_story_hir)*100,1)}   ", end = "\r")
    for l in range(len(line)):
        letter = line[l]
        unic = unicodedata.name(letter)
        # print(unic)
        if "CJK" in unic:
            kanji_list.append(letter)


72979-45377
# %%

kanji_unique = np.unique(kanji_list)


counter = collections.Counter(kanji_list)

counter_dict = dict(counter)


df = pd.DataFrame.from_dict(counter_dict.items())
df.columns = ["kanji", "frequency"]


# %%

df["relative_frequency"] = np.nan
df["meaning"] = np.nan
df["example_on"] = np.nan
df["example_kun"] = np.nan

df["example_pla_jp1"] = ""
df["example_pla_jp2"] = ""
df["example_pla_en1"] = ""
df["example_pla_en2"] = ""


df = df.sort_values(by='frequency', ascending=False)
df = df.reset_index(drop=True)
df = df.reset_index()
df["index"] = df["index"] + 1

df["relative_frequency"] = np.round(df["frequency"] / np.max(df["frequency"]) * 100, 0)
df["relative_frequency"] = pd.to_numeric(df["relative_frequency"], downcast='integer')

# %%
for i in range(1287, len(df)):
    letter = df["kanji"][i]
    print(f"\r{i}/{len(df)}; {np.round(i/len(df)*100,1)}, {letter}", end= "\r")
    k = Kanji.request(letter)
    if k is not None:
        meanings = k.data.main_meanings
        meanings = ", ".join(meanings)
        df.loc[i, "meaning"] = meanings
    if not k.data.reading_examples == None:
        if not k.data.reading_examples.on == None:
            # print(f"Onyomi")
            len_on = len(k.data.reading_examples.on)
            on_vec = []
            for o in range(len_on):
                r_kan = k.data.reading_examples.on[o].kanji
                r_read = k.data.reading_examples.on[o].reading
                r_mean = k.data.reading_examples.on[o].meanings
    
                meaning = ", ".join(r_mean)
                if o < len_on-1:
                    add = "<br>"
                else:
                    add = ""
                on_vec.append(f"{r_kan}   [{r_read}]   {meaning}{add}")
                #print(f"{i}, {r_kan}, {r_read}, {meaning}")
            df.loc[i, "example_on"] = "".join(on_vec)
    if not k.data.reading_examples == None:
        if not k.data.reading_examples.kun == None:
            len_kun = len(k.data.reading_examples.kun)
            # print(f"Kunyomi")
            kun_vec = []
            for o in range(len_kun):
                r_kan = k.data.reading_examples.kun[o].kanji
                r_read = k.data.reading_examples.kun[o].reading
                r_mean = k.data.reading_examples.kun[o].meanings
    
                meaning = ", ".join(r_mean)
                if o < len_kun-1:
                    add = "<br>"
                else:
                    add = ""
                kun_vec.append(f"{r_kan}   [{r_read}]   {meaning}{add}")
                #print(f"{i}, {r_kan}, {r_read}, {meaning}")
            df.loc[i, "example_kun"] = "".join(kun_vec)
    # Find where used

    ind_common = pd.DataFrame(columns = ["ind", "jp", "en", "length"])
    ind_story = pd.DataFrame(columns = ["ind", "jp", "en", "length"])

    for l in range(len(pla_common_hir)):
        line = pla_common_hir[l]
        en = pla_common_en[l]

        for character in remove_characters:
            line = line.replace(character, "")
            en = en.replace(character, "")
        if letter in line:
            ind_common = ind_common.append( dict( ind = l, jp = line, en = en, length = len(line)  ),ignore_index=True  )
    for l in range(len(pla_story_hir)):
        line = pla_story_hir[l]
        en = pla_story_en[l]

        for character in remove_characters:
            line = line.replace(character, "")
            en = en.replace(character, "")
        if letter in line:
            ind_story = ind_story.append( dict( ind = l, jp = line, en = en, length = len(line)  ),ignore_index=True  )
    # get unique and sort
    ind_common = ind_common.drop_duplicates('jp')
    ind_story = ind_story.drop_duplicates('jp')
    ind_common = ind_common.sort_values("length").reset_index()
    ind_story = ind_story.sort_values("length").reset_index()


    if len(ind_common) > 1:
        #get shortest examples
        if i == 0:
            ind = [6,4]
        else:
            ind = [0,1]
        example1_jp = ind_common.loc[ind[0], "jp"]
        example2_jp =ind_common.loc[ind[1], "jp"]
        example1_en = ind_common.loc[ind[0], "en"]
        example2_en = ind_common.loc[ind[1], "en"]



    elif len(ind_common) == 1:
        example1_jp = ind_common.loc[ind[0], "jp"]
        example2_jp = ""
        example1_en = ind_common.loc[ind[0], "en"]
        example2_en = ""
    elif len(ind_story) > 1:
        ind = [0,1]
        example1_jp = ind_story.loc[ind[0], "jp"]
        example2_jp =ind_story.loc[ind[1], "jp"]
        example1_en = ind_story.loc[ind[0], "en"]
        example2_en = ind_story.loc[ind[1], "en"]
    elif len(ind_story) == 1:
        example1_jp = ind_story.loc[ind[0], "jp"]
        example2_jp = ""
        example1_en = ind_story.loc[ind[0], "en"]
        example2_en = ""
    else:
        example1_jp = ""
        example1_en = ""
        example2_jp = ""
        example2_en = ""

    df.loc[i, "example_pla_jp1"] = re.sub("[\[]VAR.*?[\]]", " ___ ", example1_jp)
    df.loc[i, "example_pla_jp2"] = re.sub("[\[]VAR.*?[\]]", " ___ ", example2_jp)
    df.loc[i, "example_pla_en1"] = re.sub("[\[]VAR.*?[\]]", " ___ ", example1_en)
    df.loc[i, "example_pla_en2"] = re.sub("[\[]VAR.*?[\]]", " ___ ", example2_en)



# %%

df.to_csv("text/output/PLA_kanji.txt", sep="\t", header=None, index=None)

# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================
# %% Sentences and vocab


pla_common_hir = pd.read_csv(path_pla_common_hir, encoding='UTF-16', sep="\n")
pla_common_kat = pd.read_csv(path_pla_common_kat, encoding='UTF-16', sep="\n")
pla_story_hir = pd.read_csv(path_pla_story_hir, encoding='UTF-16', sep="\n")
pla_story_kat = pd.read_csv(path_pla_story_kat, encoding='UTF-16', sep="\n")
pla_common_en = pd.read_csv(path_pla_common_en, encoding='UTF-16', sep="\n")
pla_story_en = pd.read_csv(path_pla_story_en, encoding='UTF-16', sep="\n")


pla_common_hir = pla_common_hir.fillna('')
pla_story_hir = pla_story_hir.fillna('')


with open(path_pla_common_hir, encoding='utf16') as f:
    pla_common_hir = f.readlines()
with open(path_pla_common_kat, encoding='utf16') as f:
    pla_common_kat = f.readlines()
with open(path_pla_story_hir, encoding='utf16') as f:
    pla_story_hir = f.readlines()
with open(path_pla_story_kat, encoding='utf16') as f:
    pla_story_kat = f.readlines()
with open(path_pla_common_en, encoding='utf16') as f:
    pla_common_en = f.readlines()
with open(path_pla_story_en, encoding='utf16') as f:
    pla_story_en = f.readlines()

columns_dialogue = columns=["dialogue", "furigana", "kana", "english", "vocab",
                        "kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"]




# %%

#dialogue ,pla_story_hir, pla_story_en
def get_info(pla_story_hir_var, pla_story_en_var, dialogue_var, start = 0 , stop = None, skip = []):
    if stop == None:
        stop = len(pla_story_hir_var)
    for i in range(start, stop):
        print(f"\n\n{i}/{len(pla_story_hir_var)}; {np.round(i/len(pla_story_hir_var)*100,1)}")
        line = pla_story_hir_var[i]
        #if ignore == 1:

        if i in skip:
            continue

    

        for character in remove_characters:
            line = line.replace(character, "")
    
        contains_japanse = []
    
        for l in range(len(line)):
            letter = line[l]
            unic = unicodedata.name(letter)
            # print(unic)
            if "CJK" in unic or "HIRAGANA" in unic or "KATAKANA" in unic:
                contains_japanse.append(True)
            else:
                contains_japanse.append(False)
        if any(contains_japanse):
            line_jp = re.sub("[\[]VAR.*?[\]]", " ___ ", line)
            #line_jp = re.sub("[\[]WAIT.*?[\]]", "", line)
    
    
    
            if any( dialogue_var['dialogue'].str.contains(line_jp)): #if already has an entry
                continue
    
            eng = re.sub("[\[]VAR.*?[\]]", " ___ ", pla_story_en_var[i])
            for character in remove_characters:
                eng = eng.replace(character, "")
            print(f"{line_jp}\n{eng}")
    
            kanji_entry = ""
            count = 0
            for l in range(len(line_jp)):
                print(f"\rKanji: {np.round((l+1)/len(line_jp)*100,1)}%   ", end='\r')
                letter = line_jp[l]
                unic = unicodedata.name(letter)
                if "CJK" in unic:
                    k = Kanji.request(letter)
                    if k is not None:
                        meanings = k.data.main_meanings
                        meanings = ", ".join(meanings)
                        if count == 0:
                            start = ""
                        else:
                            start = "<br>"
                        input_string = f"{start}<b>{letter}</b>: {meanings}"
                        kanji_entry = kanji_entry + input_string
                        count = count + 1
    
            tokens = Tokens.request(line_jp)
    
    
    
    
            # get furigana
            urllib.parse.quote(line_jp)
            url = URL + urllib.parse.quote(line_jp)
            r = requests.get(url).content
            soup = BeautifulSoup(r, "html.parser")
            res = soup.find_all("section", {"id": "zen_bar"})
            tks = []
            for r in res:
                toks = r.find_all("li")
                for t in toks:
                    #t = toks[1]
                    try:
                        pos_tag = t['data-pos']
                    except:
                        pos_tag = "Unknown"
                    jp = t.find_all('span', {"class": "japanese_word__furigana"})
                    if len(jp) > 0:
                        furi = ""
                        for s in jp:
                            #s = jp[0]
                            if len(s['data-text']) == 0:
                                furi = f"{furi}{s.text.strip()} "
                            else:
                                contains_kanji = []
                                for l in range(len(s['data-text'])):
                                    letter = s['data-text'][l]
                                    unic = unicodedata.name(letter)
                                    # print(unic)
                                    if "CJK" in unic:
                                        contains_kanji.append(True)
                                    else:
                                        contains_kanji.append(False)
                                if any(contains_kanji):
                                    furi = f"{furi}{s['data-text']}[{s.text.strip()}] "
                                else:
                                    furi = f"{furi}{s['data-text']} "
                    else:
                        try:
                            jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                            furi = jp[0].find_all('a')[0]['data-word']
                        except Exception as e:
                            jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                            furi = jp[0].text.strip()
                    tks.append(TokenConfig(
                        token=furi,
                        pos_tag=pos_tag
                    ))
    
            tks_kana = []
            for r in res:
    
                toks = r.find_all("li")
                for t in toks:
                    # t = toks[5]
                    try:
                        pos_tag = t['data-pos']
                    except:
                        pos_tag = "Unknown"
                    jp = t.find_all('span', {"class": "japanese_word__furigana"})
                    if len(jp) > 0:
                        furi = ""
                        for s in jp:
                            #s = jp[0]
                            if len(s['data-text']) == 0:
                                furi = f"{furi}{s.text.strip()}"
                            else:
                                contains_kanji = []
                                for l in range(len(s['data-text'])):
                                    letter = s['data-text'][l]
                                    unic = unicodedata.name(letter)
                                    # print(unic)
                                    if "CJK" in unic:
                                        contains_kanji.append(True)
                                    else:
                                        contains_kanji.append(False)
                                if any(contains_kanji):
                                    furi = f"{furi}{s.text.strip()}"
                                else:
                                    furi = f"{furi}{s['data-text']}"
                    else:
                        try:
                            jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                            furi = jp[0].find_all('a')[0]['data-word']
                        except Exception as e:
                            jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                            furi = jp[0].text.strip()
                    tks_kana.append(TokenConfig(
                        token=furi,
                        pos_tag=pos_tag
                    ))
    
            tokens
            tks
            tks_kana
    
            kana = ""
            kana_style = ""
            furigana = ""
            furigana_style = ""
            kanji_style = ""
            vocab = []
            vocab_style = []
            allowed_vocab = ["Noun", "Verb", "Adjective", "Adverb"]
    
            # colors = {"Noun": "red", "Particle": "#d2cdbf", "Verb": "yellow", "Adjective": "green", "Pronoun": "purple", "Proper noun": "pink", "Adverb": "rd", "Conjunction": "", "Determiner": "", "Prefix": "" , "Suffix": "","Interjection": "", "Unknown": ""}
            colors = {"Noun": "#C4D4F5", "Particle": "#d2cdbf", "Verb": "#d2bfc4",
                      "Adjective": "#c4d2bf", "Pronoun": "#dbd5f8", "Adverb": "#edd5f8"}
            if not tokens == None:
                for k in range(len(tokens)):
                    input_word = tokens.data[k].token
                    input_pos = tokens.data[k].pos_tag
                    input_word_furi = tks[k].token
                    input_word_kana = tks_kana[k].token
        
                    # is there a space after word
        
                    ind = line_jp.find(input_word) + len(input_word)
                    if ind < len(line_jp):
                        if line_jp[ind] == " ":
                            space = " "
                        else:
                            space = ""
                    else:
                        space = ""
        
                    if input_pos.value in colors:
                        font_col = colors[input_pos.value]
        
                        font_start = f'<font color="{font_col}">'
                        font_end = "</font>"
                    else:
                        font_start = ''
                        font_end = ""
        
                    kana = f"{kana}{input_word_kana}{space}"
                    furigana = f"{furigana}{input_word_furi}{space}"
        
                    kana_style = f"{kana_style}{font_start}{input_word_kana}{font_end}{space}"
                    furigana_style = f"{furigana_style}{font_start}{input_word_furi}{font_end}{space}"
                    kanji_style = f"{kanji_style}{font_start}{input_word}{font_end}{space}"
    
                # get vocab words
    
                for k in range(len(tokens)):
                    print(f"\rVocab: {np.round((k+1)/len(tokens)*100,1)}%  ", end='\r')
                    input_word = tokens.data[k].token
                    input_pos = tokens.data[k].pos_tag
                    input_word_furi = tks[k].token
                    input_word_kana = tks_kana[k].token
                    if input_pos.value in colors:
                        font_col = colors[input_pos.value]
        
                        font_start = f'<font color="{font_col}">'
                        font_end = "</font>"
                    else:
                        font_col = "#ffffed"
                        font_start = '<font color="{font_col}">'
                        font_end = "</font>"
        
                    if input_pos.value in allowed_vocab:
                        word = Word.request(input_word)
                        time.sleep(0.45)
                        if not word == None:
                            definition = ", ".join(word.data[0].senses[0].english_definitions)
                            add = "<br>"
                            contains_kanji = []
                            for l in range(len(input_word)):
                                letter = input_word[l]
                                unic = unicodedata.name(letter)
                                # print(unic)
                                if "CJK" in unic:
                                    contains_kanji.append(True)
                                else:
                                    contains_kanji.append(False)
                            if any(contains_kanji):
                                vocab_style.append(
                                    f'{font_start}{input_word_furi}{font_end}:   {definition}{add}')
                            else:
                                vocab_style.append(
                                    f"{font_start}{input_word_furi}{font_end}:    {definition}{add}")
                            if any(contains_kanji):
                                vocab.append(f'{input_word_furi}:   {definition}{add}')
                            else:
                                vocab.append(f"{input_word_furi}:    {definition}{add}")
    
            if tokens == None:
    
                soup.find_all("section")
                soup.prettify()
                if not len(soup.find_all('span', {"class": "furigana"})) == 0:
                    furis = soup.find_all('span', {"class": "furigana"})[0].find_all('span')
        
    
                    coun = 0
                    for l in range(len(line_jp)):
                        print(f"\rKanji: {np.round((l+1)/len(line_jp)*100,1)}%   ", end='\r')
                        letter = line_jp[l]
                        unic = unicodedata.name(letter)
                        if "CJK" in unic:
                            k =  furis[coun].text.strip()
                            furigana = f"{furigana}{letter}[{k}] "
                            kana = f"{kana}{k}"
                            coun = coun+1
                        else:
                            furigana = f"{furigana}{letter} "
                            kana = f"{kana}{letter}"
                else:
    
                    for l in range(len(line_jp)):
                        print(f"\rKanji: {np.round((l+1)/len(line_jp)*100,1)}%   ", end='\r')
                        letter = line_jp[l]
                        unic = unicodedata.name(letter)
                        if "CJK" in unic:
                            k = Kanji.request(letter)
                            try:
                                input_k = k.data.main_readings.kun[0].split('.')[0]
                            except:
                                input_k = k.data.main_readings.on[0].split('.')[0]
                            furigana = f"{furigana}{letter}[{input_k}] "
                            kana = f"{kana}{input_k}"
                        else:
                            furigana = f"{furigana}{letter} "
                            kana = f"{kana}{letter}"
                    word = Word.request(line_jp)
                    if not word == None:
                        kana = word.data[0].japanese[0].reading
                kanji_style = line_jp
                furigana_style = furigana
                kana_style = kana
    
            _, idx = np.unique(vocab, return_index=True)
            vocab = "".join(list(np.array(vocab)[np.sort(idx)]))[:-4]
            vocab_style = "".join(list(np.array(vocab_style)[np.sort(idx)]))[:-4]
    
            dialogue_var =  dialogue_var.append(dict(dialogue=line_jp, furigana=furigana,  kana=kana,  english=eng,  vocab=vocab,  kanji=kanji_entry,
                                       kanji_style=kanji_style,  furigana_style=furigana_style,  kana_style=kana_style, vocab_style=vocab_style), ignore_index=True)


    return dialogue_var


# %%

dialogue = pd.DataFrame(columns=["dialogue", "furigana", "kana", "english", "vocab",
                        "kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"])


dialogue = pd.read_csv("text/output/PLA_dialogue1.txt", sep="\t", header=None)
dialogue.columns = ["index"] + columns_dialogue
dialogue = dialogue.drop("index", axis = 1)

#%%
skip1 = [1701,7570]
for i in range(0, len(pla_story_hir)):
    try:
        dialogue = get_info(pla_story_hir, pla_story_en, dialogue, start = i , stop = i+1,  skip = skip1)
    except:
        try:
            dialogue = get_info(pla_story_hir, pla_story_en, dialogue, start = i , stop = i+1,skip = skip1)
        except:

            dialogue = get_info(pla_story_hir, pla_story_en, dialogue, start = i , stop = i+1,skip = skip1)
            print(i)






#%%

dialogue = dialogue.drop_duplicates()
dialogue_index = copy.deepcopy(dialogue)
dialogue_index = dialogue_index.reset_index()
# %%
dialogue_index.to_csv("text/output/PLA_dialogue1.txt", sep="\t", header=None, index=None)


# %% remove weird blanks
dialogue_blanks = pd.read_csv("text/output/PLA_dialogue1.txt", sep="\t", header=None)
dialogue_blanks.columns = ["index"] + columns_dialogue
dialogue_blanks = dialogue_blanks.drop("index", axis = 1)

for d in range(len(dialogue_blanks)):
    print(f"\r{d+1}/{len(dialogue_blanks)}; {np.round((d+1)/len(dialogue_blanks)*100,1)}%     ",end = "\r")
    jp = dialogue_blanks.loc[d, "dialogue"]
    fu = dialogue_blanks.loc[d, "furigana"]
    kana = dialogue_blanks.loc[d, "kana"]
    kanji_st = dialogue_blanks.loc[d, "kanji_style"]
    fu_st = dialogue_blanks.loc[d, "furigana_style"]
    kana_st = dialogue_blanks.loc[d, "kana_style"]
    en= dialogue_blanks.loc[d, "english"]

    if " ___ isare  ___  wisp ___ s "in en:
        en = en.replace(" ___ isare  ___  wisp ___ s ", "is/are  ___  wisp(s)")
    if "wisp ___ s"in en:
        en = en.replace("wisp ___ s", "wisp(s)")


    jp = jp.replace(" ___", "")
    fu = fu.replace("___", "")
    kana = kana.replace("___", "")
    kanji_st = fu_st.replace("___", "")
    fu_st = fu_st.replace("___", "")
    kana_st = kana_st.replace("___", "")
    en = en.replace(" ___ ", "")

    dialogue_blanks.loc[d, "dialogue"]= jp
    dialogue_blanks.loc[d, "furigana"] = fu
    dialogue_blanks.loc[d, "kana"] = kana
    dialogue_blanks.loc[d, "kanji_style"] = kanji_st
    dialogue_blanks.loc[d, "furigana_style"] = fu_st
    dialogue_blanks.loc[d, "kana_style"] = kana_st
    dialogue_blanks.loc[d, "english"] = en

    #swtich sinnoh
    shinnoh = 'シンオウ:    to do, to carry out, to perform'
    shinnoh_style = 'シンオウ</font>:    to do, to carry out, to perform'
    shinnoh_replace = "シンオウ:    Sinnoh"
    shinnoh_style_replace = "シンオウ</font>:    Sinnoh"
    if isinstance( dialogue_blanks.loc[d, "vocab"], str):
        if shinnoh in dialogue_blanks.loc[d, "vocab"]:
            print("changing Sinnoh")
            dialogue_blanks.loc[d, "vocab"] = dialogue_blanks.loc[d, "vocab"].replace(shinnoh, shinnoh_replace)
        if shinnoh_style in dialogue_blanks.loc[d, "vocab_style"]:
            dialogue_blanks.loc[d, "vocab_style"] = dialogue_blanks.loc[d, "vocab_style"].replace(shinnoh_style, shinnoh_style_replace)

    #swtich Hisui
    shinnoh = 'ヒスイ:    Jade'
    shinnoh_style = 'ヒスイ</font>:    Jade'
    shinnoh_replace = "ヒスイ:    Hisui"
    shinnoh_style_replace = "ヒスイ</font>:    Hisui"
    if isinstance( dialogue_blanks.loc[d, "vocab"], str):
        if shinnoh in dialogue_blanks.loc[d, "vocab"]:
            print("changing Hisui")
            dialogue_blanks.loc[d, "vocab"] = dialogue_blanks.loc[d, "vocab"].replace(shinnoh, shinnoh_replace)
        if shinnoh_style in dialogue_blanks.loc[d, "vocab_style"]:
            dialogue_blanks.loc[d, "vocab_style"] = dialogue_blanks.loc[d, "vocab_style"].replace(shinnoh_style, shinnoh_style_replace)


dialogue_blanks_index = copy.deepcopy(dialogue_blanks)
dialogue_blanks_index = dialogue_blanks_index.reset_index()
# %%
dialogue_blanks_index.to_csv("text/output/PLA_dialogue1_blanks_removed.txt", sep="\t", header=None, index=None)

#%%

# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================

# %% dialogue2
dialogue2 = pd.DataFrame(columns=["dialogue", "furigana", "kana", "english", "vocab",
                        "kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"])



dialogue2 = pd.read_csv("text/output/PLA_dialogue2.txt", sep="\t", header=None)
dialogue2.columns = ["index"] + columns_dialogue
dialogue2 = dialogue2.drop("index", axis = 1)

# %% dialogue2

skip2 = [681, 2935,3217, 3235, 4129, 11768, 21061,  23141, 23142, 23173, 23197, 23848,23849, 23850, 23851]
for i in range(0, len(pla_common_hir)):
    try:
        dialogue2 = get_info(pla_story_hir_var = pla_common_hir, pla_story_en_var = pla_common_en, dialogue_var = dialogue2, start = i , stop = i+1, skip = skip2)
    except:
        try:
            dialogue2 = get_info(pla_common_hir, pla_common_en, dialogue2, start = i , stop = i+1,   skip = skip2)
        except:
            try:
                dialogue2 = get_info(pla_common_hir, pla_common_en, dialogue2, start = i , stop = i+1,   skip = skip2)
            except:
                continue




#%%

dialogue2 = dialogue2.drop_duplicates()
dialogue2_index = copy.deepcopy(dialogue2)
dialogue2_index = dialogue2_index.reset_index()
# %%
dialogue2_index.to_csv("text/output/PLA_dialogue2.txt", sep="\t", header=None, index=None)

# %%

dialogue_blanks = pd.read_csv("text/output/PLA_dialogue2.txt", sep="\t", header=None)
dialogue_blanks.columns = ["index"] + columns_dialogue
dialogue_blanks = dialogue_blanks.drop("index", axis = 1)

for d in range(len(dialogue_blanks)):
    print(f"\r{d+1}/{len(dialogue_blanks)}; {np.round((d+1)/len(dialogue_blanks)*100,1)}%     ",end = "\r")
    jp = dialogue_blanks.loc[d, "dialogue"]
    fu = dialogue_blanks.loc[d, "furigana"]
    kana = dialogue_blanks.loc[d, "kana"]
    kanji_st = dialogue_blanks.loc[d, "kanji_style"]
    fu_st = dialogue_blanks.loc[d, "furigana_style"]
    kana_st = dialogue_blanks.loc[d, "kana_style"]
    en= dialogue_blanks.loc[d, "english"]


    #swtich sinnoh
    shinnoh = 'シンオウ:    to do, to carry out, to perform'
    shinnoh_style = 'シンオウ</font>:    to do, to carry out, to perform'
    shinnoh_replace = "シンオウ:    Sinnoh"
    shinnoh_style_replace = "シンオウ</font>:    Sinnoh"
    if isinstance( dialogue_blanks.loc[d, "vocab"], str):
        if shinnoh in dialogue_blanks.loc[d, "vocab"]:
            print("changing Sinnoh     ")
            dialogue_blanks.loc[d, "vocab"] = dialogue_blanks.loc[d, "vocab"].replace(shinnoh, shinnoh_replace)
        if shinnoh_style in dialogue_blanks.loc[d, "vocab_style"]:
            dialogue_blanks.loc[d, "vocab_style"] = dialogue_blanks.loc[d, "vocab_style"].replace(shinnoh_style, shinnoh_style_replace)
    #swtich Hisui
    shinnoh = 'ヒスイ:    Jade'
    shinnoh_style = 'ヒスイ</font>:    Jade'
    shinnoh_replace = "ヒスイ:    Hisui"
    shinnoh_style_replace = "ヒスイ</font>:    Hisui"
    if isinstance( dialogue_blanks.loc[d, "vocab"], str):
        if shinnoh in dialogue_blanks.loc[d, "vocab"]:
            print("changing Hisui")
            dialogue_blanks.loc[d, "vocab"] = dialogue_blanks.loc[d, "vocab"].replace(shinnoh, shinnoh_replace)
        if shinnoh_style in dialogue_blanks.loc[d, "vocab_style"]:
            dialogue_blanks.loc[d, "vocab_style"] = dialogue_blanks.loc[d, "vocab_style"].replace(shinnoh_style, shinnoh_style_replace)


dialogue_blanks_index = copy.deepcopy(dialogue_blanks)
dialogue_blanks_index = dialogue_blanks_index.reset_index()
# %%
dialogue_blanks_index.to_csv("text/output/PLA_dialogue2_blanks_removed.txt", sep="\t", header=None, index=None)


# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================
# =============================================================================
#
# =============================================================================



#%%


dia1 = pd.read_csv("text/output/PLA_dialogue1_blanks_removed.txt", sep="\t", header=None)
dia2 = pd.read_csv("text/output/PLA_dialogue2_blanks_removed.txt", sep="\t", header=None)


dia2.columns= ["index", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"]
dia1.columns= ["index", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"]
# %%




df_vocab2 = pd.DataFrame(columns = ["word", "word_furigana", "word_english", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "dialogue_length"])
# Get vocab
for i in range(len(dia2)):
    print(f"\r{i}/{len(dia2)}; {np.round(i/len(dia2)*100,1)}        ", end = "\r")

    vocabs  =dia2.loc[i, "vocab"]
    if isinstance(vocabs, str):
        splits = vocabs.split("<br>")
        for f in range(len(splits)):
            vo = splits[f]
            word_furigana, word_english = vo.split(":",1)
            word_furigana = word_furigana.strip()
            word_english = word_english.strip()

            word = re.sub("[\[].*?[\]]", "", word_furigana)
            word = word.replace(" ", "")
            input_entry = dict(word = word,
                 word_furigana = word_furigana,
                 word_english = word_english,
                 dialogue = dia2.loc[i, "dialogue"],
                 furigana = dia2.loc[i, "furigana"],
                 kana = dia2.loc[i, "kana"],
                 english =dia2.loc[i, "english"] ,
                 vocab = dia2.loc[i, "vocab"],
                 kanji = dia2.loc[i, "kanji"],
                 kanji_style = dia2.loc[i, "kanji_style"],
                 furigana_style = dia2.loc[i, "furigana_style"],
                 kana_style = dia2.loc[i, "kana_style"],
                 vocab_style  = dia2.loc[i, "vocab_style"] ,
                 dialogue_length = len(dia2.loc[i, "dialogue"]))
            df_vocab2 =  df_vocab2.append(input_entry, ignore_index=True  )
    elif math.isnan(vocabs):

        word_furigana, word_english = dia2.loc[i, "furigana"],  dia2.loc[i, "english"]
        word_furigana = word_furigana.strip()
        word_english = word_english.strip()


        word = dia2.loc[i, "dialogue"]
        word = word.replace(" ", "")


        input_entry = dict(word = word,
             word_furigana = word_furigana,
             word_english = word_english,
             dialogue = dia2.loc[i, "dialogue"],
             furigana = dia2.loc[i, "furigana"],
             kana = dia2.loc[i, "kana"],
             english =dia2.loc[i, "english"] ,
             vocab = dia2.loc[i, "vocab"],
             kanji = dia2.loc[i, "kanji"],
             kanji_style = dia2.loc[i, "kanji_style"],
             furigana_style = dia2.loc[i, "furigana_style"],
             kana_style = dia2.loc[i, "kana_style"],
             vocab_style  = dia2.loc[i, "vocab_style"] ,
             dialogue_length = len(dia2.loc[i, "dialogue"]))
        df_vocab2 =  df_vocab2.append(input_entry, ignore_index=True  )

#%%
df_vocab2_index =  df_vocab2.reset_index()

df_vocab2_index_sort =  df_vocab2_index.sort_values("dialogue_length")
df_vocab2_index_sort_unique =  df_vocab2_index_sort.drop_duplicates("word", keep = "first")


df_vocab2_unique =  df_vocab2_index_sort_unique.sort_values("index")

df_vocab2_unique.loc[ df_vocab2_unique["word_english"] == "Jade", "word_english"] = "Hisui"
df_vocab2_unique.loc[ df_vocab2_unique["word_english"] == "American", "word_english"] = "Candy"
df_vocab2_unique=df_vocab2_unique.rename(columns = {'index':'index_by_dialogue'})

df_vocab2_unique = df_vocab2_unique.reset_index(drop=True)

counter_vocab = collections.Counter(df_vocab2_index["word"])

counter_dict = dict(counter_vocab)

df_vocab2_unique["frequency"] = np.nan
for i in range(len(df_vocab2_unique)):
    word = df_vocab2_unique.loc[i, "word"]
    df_vocab2_unique.loc[i, "frequency"] = counter_dict[word]

df_vocab2_unique["frequency"] = pd.to_numeric(df_vocab2_unique["frequency"], downcast='integer')

df_vocab2_unique_sort_freq =  df_vocab2_unique.sort_values("frequency", ascending = False)


df_vocab2_save = df_vocab2_unique_sort_freq.reset_index(drop=True)
df_vocab2_save = df_vocab2_save.reset_index()
df_vocab2_save = df_vocab2_save.drop("dialogue_length", axis = 1)
# %%
def swap_columns(df, c1, c2):
    df = copy.deepcopy(df)
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)
    df.columns
    df.rename(columns={c1:'temp'}, inplace=True)
    df.rename(columns={c2:c1}, inplace=True)
    df.rename(columns={'temp':c2}, inplace=True)
    return df


df_vocab2_save = swap_columns(df = df_vocab2_save, c1 = 'index_by_dialogue', c2 = 'frequency')


df_vocab2_save.to_csv("text/output/PLA_vocab2.txt", sep="\t", header=None, index=None)



























# %% Vocab 1


# =============================================================================
#
# =============================================================================

dia1 = pd.read_csv("text/output/PLA_dialogue1_blanks_removed.txt", sep="\t", header=None)
dia2 = pd.read_csv("text/output/PLA_dialogue2_blanks_removed.txt", sep="\t", header=None)


dia2.columns= ["index", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"]
dia1.columns= ["index", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"]

# =============================================================================
#
# =============================================================================



# =============================================================================
#
# =============================================================================


# =============================================================================
#
# =============================================================================
# %%

df_vocab1 = pd.DataFrame(columns = ["word", "word_furigana", "word_english", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "dialogue_length"])
# Get vocab
for i in range(len(dia1)):
    print(f"\r{i+1}/{len(dia1)}; {np.round((i+1)/len(dia1)*100,1)}        ", end = "\r")

    vocabs  =dia1.loc[i, "vocab"]
    if isinstance(vocabs, str):
        splits = vocabs.split("<br>")
        for f in range(len(splits)):
            vo = splits[f]
            word_furigana, word_english = vo.split(":",1)
            word_furigana = word_furigana.strip()
            word_english = word_english.strip()

            word = re.sub("[\[].*?[\]]", "", word_furigana)
            word = word.replace(" ", "")
            input_entry = dict(word = word,
                 word_furigana = word_furigana,
                 word_english = word_english,
                 dialogue = dia1.loc[i, "dialogue"],
                 furigana = dia1.loc[i, "furigana"],
                 kana = dia1.loc[i, "kana"],
                 english =dia1.loc[i, "english"] ,
                 vocab = dia1.loc[i, "vocab"],
                 kanji = dia1.loc[i, "kanji"],
                 kanji_style = dia1.loc[i, "kanji_style"],
                 furigana_style = dia1.loc[i, "furigana_style"],
                 kana_style = dia1.loc[i, "kana_style"],
                 vocab_style  = dia1.loc[i, "vocab_style"] ,
                 dialogue_length = len(dia1.loc[i, "dialogue"]))
            df_vocab1 =  df_vocab1.append(input_entry, ignore_index=True  )
    elif math.isnan(vocabs):

        word_furigana, word_english = dia1.loc[i, "furigana"],  dia1.loc[i, "english"]
        word_furigana = word_furigana.strip()
        word_english = word_english.strip()


        word = dia1.loc[i, "dialogue"]
        word = word.replace(" ", "")


        input_entry = dict(word = word,
             word_furigana = word_furigana,
             word_english = word_english,
             dialogue = dia1.loc[i, "dialogue"],
             furigana = dia1.loc[i, "furigana"],
             kana = dia1.loc[i, "kana"],
             english =dia1.loc[i, "english"] ,
             vocab = dia1.loc[i, "vocab"],
             kanji = dia1.loc[i, "kanji"],
             kanji_style = dia1.loc[i, "kanji_style"],
             furigana_style = dia1.loc[i, "furigana_style"],
             kana_style = dia1.loc[i, "kana_style"],
             vocab_style  = dia1.loc[i, "vocab_style"] ,
             dialogue_length = len(dia1.loc[i, "dialogue"]))
        df_vocab1 =  df_vocab1.append(input_entry, ignore_index=True  )

#%%
df_vocab1_index =  df_vocab1.reset_index()

df_vocab1_index_sort =  df_vocab1_index.sort_values("dialogue_length")
df_vocab1_index_sort_unique =  df_vocab1_index_sort.drop_duplicates("word", keep = "first")


df_vocab1_unique =  df_vocab1_index_sort_unique.sort_values("index")

df_vocab1_unique.loc[ df_vocab1_unique["word_english"] == "Jade", "word_english"] = "Hisui"
df_vocab1_unique.loc[ df_vocab1_unique["word_english"] == "American", "word_english"] = "Candy"
df_vocab1_unique=df_vocab1_unique.rename(columns = {'index':'index_by_dialogue'})

df_vocab1_unique = df_vocab1_unique.reset_index(drop=True)

counter_vocab = collections.Counter(df_vocab1_index["word"])

counter_dict = dict(counter_vocab)

df_vocab1_unique["frequency"] = np.nan
for i in range(len(df_vocab1_unique)):
    word = df_vocab1_unique.loc[i, "word"]
    df_vocab1_unique.loc[i, "frequency"] = counter_dict[word]

df_vocab1_unique["frequency"] = pd.to_numeric(df_vocab1_unique["frequency"], downcast='integer')

df_vocab1_unique_sort_freq =  df_vocab1_unique.sort_values("frequency", ascending = False)


df_vocab1_save = df_vocab1_unique_sort_freq.reset_index(drop=True)
df_vocab1_save = df_vocab1_save.reset_index()
df_vocab1_save = df_vocab1_save.drop("dialogue_length", axis = 1)

# %%
#Changing the first entry for "no"

x = 2363
ind = np.where(df_vocab1_save["word"] == "の")[0][0]


df_vocab1_save.loc[ind,"dialogue"] = dia1.loc[x, "dialogue"]
df_vocab1_save.loc[ind,"furigana"] = dia1.loc[x, "furigana"]
df_vocab1_save.loc[ind,"kana"] = dia1.loc[x, "kana"]
df_vocab1_save.loc[ind,"english"] = dia1.loc[x, "english"]
df_vocab1_save.loc[ind,"vocab"] = dia1.loc[x, "vocab"]
df_vocab1_save.loc[ind,"kanji"] = dia1.loc[x, "kanji"]
df_vocab1_save.loc[ind,"kanji_style"] = dia1.loc[x, "kanji_style"]
df_vocab1_save.loc[ind,"furigana_style"] = dia1.loc[x, "furigana_style"]
df_vocab1_save.loc[ind,"kana_style"] = dia1.loc[x, "kana_style"]
df_vocab1_save.loc[ind,"vocab_style"] =dia1.loc[x, "vocab_style"]

# %%

ind = np.where(df_vocab1_save["word"] == "で")[0][0]

df_vocab1_save.loc[ind, "word"] = "である"
df_vocab1_save.loc[ind, "word_furigana"] = "である"
df_vocab1_save.loc[ind, "word_english"] = "be, is"
df_vocab1_save.loc[ind, "vocab"]

dearu = "で:    coming out, going out, outflow, efflux, rising (of the Sun or the Moon)"
dearu_style = "で</font>:    coming out, going out, outflow, efflux, rising (of the Sun or the Moon)"
dearu_replace = "である:    be, is"
dearu_replace_style = "である</font>:    be, is"
df_vocab1_save.loc[ind, "vocab"] = df_vocab1_save.loc[ind, "vocab"].replace(dearu, dearu_replace)
df_vocab1_save.loc[ind, "vocab_style"] = df_vocab1_save.loc[ind, "vocab_style"].replace(dearu_style, dearu_replace_style)

df_vocab1_save.loc[ind]
# %%
def swap_columns(df, c1, c2):
    df = copy.deepcopy(df)
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)
    df.columns
    df.rename(columns={c1:'temp'}, inplace=True)
    df.rename(columns={c2:c1}, inplace=True)
    df.rename(columns={'temp':c2}, inplace=True)
    return df


df_vocab1_save = swap_columns(df = df_vocab1_save, c1 = 'index_by_dialogue', c2 = 'frequency')


df_vocab1_save.to_csv("text/output/PLA_vocab1.txt", sep="\t", header=None, index=None)




# %%






# =============================================================================
#
# =============================================================================



# =============================================================================
#
# =============================================================================


# =============================================================================
#
# =============================================================================











# %%
#dialogue2 = dialogue2.drop(index =  [1770, 1771, 1772])
for i in range(0, len(pla_common_hir)):
    print(f"\n\n{i}/{len(pla_common_hir)}; {np.round(i/len(pla_common_hir)*100,1)}")
    line = pla_common_hir[i]
    if i ==681 or i == 2935 or i == 3217 or i == 3235 or i == 4129:
        continue


    for character in remove_characters:
        line = line.replace(character, "")

    contains_japanse = []

    for l in range(len(line)):
        letter = line[l]
        unic = unicodedata.name(letter)
        # print(unic)
        if "CJK" in unic or "HIRAGANA" in unic or "KATAKANA" in unic:
            contains_japanse.append(True)
        else:
            contains_japanse.append(False)
    if any(contains_japanse):
        line_jp = re.sub("[\[]VAR.*?[\]]", " ___ ", line)
        #line_jp = re.sub("[\[]WAIT.*?[\]]", "", line)



        if any(dialogue2['dialogue'].str.contains(line_jp)): #if already has an entry
            continue

        eng = re.sub("[\[]VAR.*?[\]]", " ___ ", pla_common_en[i])
        for character in remove_characters:
            eng = eng.replace(character, "")
        print(f"{line_jp}\n{eng}")

        kanji_entry = ""
        count = 0
        for l in range(len(line_jp)):
            print(f"\rKanji: {np.round((l+1)/len(line_jp)*100,1)}%   ", end='\r')
            letter = line_jp[l]
            unic = unicodedata.name(letter)
            if "CJK" in unic:
                k = Kanji.request(letter)
                if k is not None:
                    meanings = k.data.main_meanings
                    meanings = ", ".join(meanings)
                    if count == 0:
                        start = ""
                    else:
                        start = "<br>"
                    input_string = f"{start}<b>{letter}</b>: {meanings}"
                    kanji_entry = kanji_entry + input_string
                    count = count + 1

        tokens = Tokens.request(line_jp)




        # get furigana
        urllib.parse.quote(line_jp)
        url = URL + urllib.parse.quote(line_jp)
        r = requests.get(url).content
        soup = BeautifulSoup(r, "html.parser")
        res = soup.find_all("section", {"id": "zen_bar"})
        tks = []
        for r in res:
            toks = r.find_all("li")
            for t in toks:
                #t = toks[1]
                try:
                    pos_tag = t['data-pos']
                except:
                    pos_tag = "Unknown"
                jp = t.find_all('span', {"class": "japanese_word__furigana"})
                if len(jp) > 0:
                    furi = ""
                    for s in jp:
                        #s = jp[0]
                        if len(s['data-text']) == 0:
                            furi = f"{furi}{s.text.strip()} "
                        else:
                            contains_kanji = []
                            for l in range(len(s['data-text'])):
                                letter = s['data-text'][l]
                                unic = unicodedata.name(letter)
                                # print(unic)
                                if "CJK" in unic:
                                    contains_kanji.append(True)
                                else:
                                    contains_kanji.append(False)
                            if any(contains_kanji):
                                furi = f"{furi}{s['data-text']}[{s.text.strip()}] "
                            else:
                                furi = f"{furi}{s['data-text']} "
                else:
                    try:
                        jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                        furi = jp[0].find_all('a')[0]['data-word']
                    except Exception as e:
                        jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                        furi = jp[0].text.strip()
                tks.append(TokenConfig(
                    token=furi,
                    pos_tag=pos_tag
                ))

        tks_kana = []
        for r in res:

            toks = r.find_all("li")
            for t in toks:
                # t = toks[5]
                try:
                    pos_tag = t['data-pos']
                except:
                    pos_tag = "Unknown"
                jp = t.find_all('span', {"class": "japanese_word__furigana"})
                if len(jp) > 0:
                    furi = ""
                    for s in jp:
                        #s = jp[0]
                        if len(s['data-text']) == 0:
                            furi = f"{furi}{s.text.strip()}"
                        else:
                            contains_kanji = []
                            for l in range(len(s['data-text'])):
                                letter = s['data-text'][l]
                                unic = unicodedata.name(letter)
                                # print(unic)
                                if "CJK" in unic:
                                    contains_kanji.append(True)
                                else:
                                    contains_kanji.append(False)
                            if any(contains_kanji):
                                furi = f"{furi}{s.text.strip()}"
                            else:
                                furi = f"{furi}{s['data-text']}"
                else:
                    try:
                        jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                        furi = jp[0].find_all('a')[0]['data-word']
                    except Exception as e:
                        jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                        furi = jp[0].text.strip()
                tks_kana.append(TokenConfig(
                    token=furi,
                    pos_tag=pos_tag
                ))

        tokens
        tks
        tks_kana

        kana = ""
        kana_style = ""
        furigana = ""
        furigana_style = ""
        kanji_style = ""
        vocab = []
        vocab_style = []
        allowed_vocab = ["Noun", "Verb", "Adjective", "Adverb"]

        # colors = {"Noun": "red", "Particle": "#d2cdbf", "Verb": "yellow", "Adjective": "green", "Pronoun": "purple", "Proper noun": "pink", "Adverb": "rd", "Conjunction": "", "Determiner": "", "Prefix": "" , "Suffix": "","Interjection": "", "Unknown": ""}
        colors = {"Noun": "#C4D4F5", "Particle": "#d2cdbf", "Verb": "#d2bfc4",
                  "Adjective": "#c4d2bf", "Pronoun": "#dbd5f8", "Adverb": "#edd5f8"}
        if not tokens == None:
            for k in range(len(tokens)):
                input_word = tokens.data[k].token
                input_pos = tokens.data[k].pos_tag
                input_word_furi = tks[k].token
                input_word_kana = tks_kana[k].token
    
                # is there a space after word
    
                ind = line_jp.find(input_word) + len(input_word)
                if ind < len(line_jp):
                    if line_jp[ind] == " ":
                        space = " "
                    else:
                        space = ""
                else:
                    space = ""
    
                if input_pos.value in colors:
                    font_col = colors[input_pos.value]
    
                    font_start = f'<font color="{font_col}">'
                    font_end = "</font>"
                else:
                    font_start = ''
                    font_end = ""
    
                kana = f"{kana}{input_word_kana}{space}"
                furigana = f"{furigana}{input_word_furi}{space}"
    
                kana_style = f"{kana_style}{font_start}{input_word_kana}{font_end}{space}"
                furigana_style = f"{furigana_style}{font_start}{input_word_furi}{font_end}{space}"
                kanji_style = f"{kanji_style}{font_start}{input_word}{font_end}{space}"

            # get vocab words

            for k in range(len(tokens)):
                print(f"\rVocab: {np.round((k+1)/len(tokens)*100,1)}%  ", end='\r')
                input_word = tokens.data[k].token
                input_pos = tokens.data[k].pos_tag
                input_word_furi = tks[k].token
                input_word_kana = tks_kana[k].token
                if input_pos.value in colors:
                    font_col = colors[input_pos.value]
    
                    font_start = f'<font color="{font_col}">'
                    font_end = "</font>"
                else:
                    font_col = "#ffffed"
                    font_start = '<font color="{font_col}">'
                    font_end = "</font>"
    
                if input_pos.value in allowed_vocab:
                    word = Word.request(input_word)
                    time.sleep(0.45)
                    if not word == None:
                        definition = ", ".join(word.data[0].senses[0].english_definitions)
                        add = "<br>"
                        contains_kanji = []
                        for l in range(len(input_word)):
                            letter = input_word[l]
                            unic = unicodedata.name(letter)
                            # print(unic)
                            if "CJK" in unic:
                                contains_kanji.append(True)
                            else:
                                contains_kanji.append(False)
                        if any(contains_kanji):
                            vocab_style.append(
                                f'{font_start}{input_word_furi}{font_end}:   {definition}{add}')
                        else:
                            vocab_style.append(
                                f"{font_start}{input_word_furi}{font_end}:    {definition}{add}")
                        if any(contains_kanji):
                            vocab.append(f'{input_word_furi}:   {definition}{add}')
                        else:
                            vocab.append(f"{input_word_furi}:    {definition}{add}")

        if tokens == None:

            soup.find_all("section")
            soup.prettify()
            if not len(soup.find_all('span', {"class": "furigana"})) == 0:
                furis = soup.find_all('span', {"class": "furigana"})[0].find_all('span')
    

                coun = 0
                for l in range(len(line_jp)):
                    print(f"\rKanji: {np.round((l+1)/len(line_jp)*100,1)}%   ", end='\r')
                    letter = line_jp[l]
                    unic = unicodedata.name(letter)
                    if "CJK" in unic:
                        k =  furis[coun].text.strip()
                        furigana = f"{furigana}{letter}[{k}] "
                        kana = f"{kana}{k}"
                        coun = coun+1
                    else:
                        furigana = f"{furigana}{letter} "
                        kana = f"{kana}{letter}"
            else:

                for l in range(len(line_jp)):
                    print(f"\rKanji: {np.round((l+1)/len(line_jp)*100,1)}%   ", end='\r')
                    letter = line_jp[l]
                    unic = unicodedata.name(letter)
                    if "CJK" in unic:
                        k = Kanji.request(letter)
                        try:
                            input_k = k.data.main_readings.kun[0].split('.')[0]
                        except:
                            input_k = k.data.main_readings.on[0].split('.')[0]
                        furigana = f"{furigana}{letter}[{input_k}] "
                        kana = f"{kana}{input_k}"
                    else:
                        furigana = f"{furigana}{letter} "
                        kana = f"{kana}{letter}"
                word = Word.request(line_jp)
                if not word == None:
                    kana = word.data[0].japanese[0].reading
            kanji_style = line_jp
            furigana_style = furigana
            kana_style = kana

        _, idx = np.unique(vocab, return_index=True)
        vocab = "".join(list(np.array(vocab)[np.sort(idx)]))[:-4]
        vocab_style = "".join(list(np.array(vocab_style)[np.sort(idx)]))[:-4]

        dialogue2 = dialogue2.append(dict(dialogue=line_jp, furigana=furigana,  kana=kana,  english=eng,  vocab=vocab,  kanji=kanji_entry,
                                   kanji_style=kanji_style,  furigana_style=furigana_style,  kana_style=kana_style, vocab_style=vocab_style), ignore_index=True)


# %%
# %%
dialogue2 = dialogue2.drop_duplicates()
dialogue2_index = copy.deepcopy(dialogue2)
dialogue2_index = dialogue2_index.reset_index()
# %%
dialogue2_index.to_csv("text/output/PLA_dialogue2.txt", sep="\t", header=None, index=None)


# %%










#%%










r = TokenRequest(
    **{
        "meta": {
            "status": 200,
        },
        "data": Tokens.tokens(soup),
    }
)

urllib.parse.quote(line_jp)
url = URL + urllib.parse.quote(line_jp)
r = requests.get(url).content
soup = BeautifulSoup(r, "html.parser")
res = soup.find_all("section", {"id": "zen_bar"})
tks = []
for r in res:
    r = res[0]
    toks = r.find_all("li")
    for t in toks:
        #t = toks[1]
        try:
            pos_tag = t['data-pos']
        except:
            pos_tag = "Unknown"
        jp = t.find_all('span', {"class": "japanese_word__furigana"})
        if len(jp) > 0:
            furi = ""
            for s in jp:
                #s = jp[0]
                if len(s['data-text']) == 0:
                    furi = f"{furi}{s.text.strip()}"
                else:
                    contains_kanji = []
                    for l in range(len(s['data-text'])):
                        letter = s['data-text'][l]
                        unic = unicodedata.name(letter)
                        # print(unic)
                        if "CJK" in unic:
                            contains_kanji.append(True)
                        else:
                            contains_kanji.append(False)
                    if any(contains_kanji):
                        furi = f"{furi}{s['data-text']}[{s.text.strip()}]"
                    else:
                        furi = f"{furi}{s['data-text']}"
        else:
            try:
                jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                furi = jp[0].find_all('a')[0]['data-word']
            except Exception as e:
                jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                furi = jp[0].text.strip()
        tks.append(TokenConfig(
            token=furi,
            pos_tag=pos_tag
        ))

tokens

tmp = jp[0].find_all('a')[0]
# %%


class RequestMeta(BaseModel):
    status: int


class TokenRequest(BaseModel):
    meta: RequestMeta
    data: List[TokenConfig]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        yield from self.data

    def rich_print(self):
        base = ''
        toks = ''
        for i, d in enumerate(self):
            base += CLITagger.underline(d.token) + ' '
            toks += f"{i}. {d.token} [violet][{str(d.pos_tag.value)}][/violet]\n"
        console.print(base)
        console.print(toks)


class Tokens:
    URL = "https://jisho.org/search/"
    ROOT = Path.home() / ".jisho/data/tokens/"

    @staticmethod
    def tokens(soup):
        res = soup.find_all("section", {"id": "zen_bar"})

        tks = []
        for r in res:
            toks = r.find_all("li")
            for t in toks:
                try:
                    pos_tag = t['data-pos']
                except:
                    pos_tag = "Unknown"
                jp = t.find_all('span', {"class": "japanese_word__text_wrapper"})
                try:
                    jp = jp[0].find_all('a')[0]['data-word']
                except Exception as e:
                    jp = jp[0].text.strip()
                tks.append(TokenConfig(
                    token=jp,
                    pos_tag=pos_tag
                ))

        return tks

    @staticmethod
    def request(word, cache=False):
        url = Tokens.URL + urllib.parse.quote(word)
        toggle = False

        if cache and (Tokens.ROOT / (word + ".json")).exists():
            toggle = True
            with open(Tokens.ROOT / (word + ".json"), "r") as fp:
                r = json.load(fp)
            r = TokenRequest(**r)
        else:
            r = requests.get(url).content
            soup = BeautifulSoup(r, "html.parser")

            r = TokenRequest(
                **{
                    "meta": {
                        "status": 200,
                    },
                    "data": Tokens.tokens(soup),
                }
            )
            if not len(r):
                console.print(f"[red bold][Error] [white] No matches found for {word}.")
                return None
        if cache and not toggle:
            Tokens.save(word, r)
        return r

    @staticmethod
    def save(word, r):
        Tokens.ROOT.mkdir(exist_ok=True)
        with open(Tokens.ROOT / f"{word}.json", "w") as fp:
            json.dump(r.dict(), fp, indent=4, ensure_ascii=False)
