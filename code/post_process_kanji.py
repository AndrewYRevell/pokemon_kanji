#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 07:54:32 2022

@author: arevell
"""

from pathlib import Path
import pprint
import json
from typing import List
import copy
import time
import pandas as pd
import os
from os.path import join
import unicodedata  # to detect if kanji or kana
import re
import numpy as np
import collections
import math
from jisho_api.kanji import Kanji
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

columns_dialogue = columns=["dialogue", "furigana", "kana", "english", "vocab",
                        "kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style"]



vocab_story = pd.read_csv("text/output/PLA_vocab1.txt", sep="\t", header=None)
vocab_common = pd.read_csv("text/output/PLA_vocab2.txt", sep="\t", header=None)

columns = ["index", "frequency", "word", "word_furigana", "word_english", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "dialogue_index", "tags"]
vocab_story.columns =columns
vocab_common.columns =columns

# %%
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
for i in range(0, len(df)):
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
    columns = ["index", "frequency", "word", "word_furigana", "word_english", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "dialogue_index", "tags"]
    for l in range(len(vocab_common)):
        line = vocab_common.loc[l,"word"]
        en = vocab_common.loc[l,"word_english"]

        if letter in line:
            ind_common = ind_common.append( dict( ind = l, jp = vocab_common.loc[l,"word_furigana"], en = en, length =  vocab_common.loc[l,"frequency"]),ignore_index=True  )


    for l in range(len(vocab_story)):
        line = vocab_story.loc[l,"word"]
        en = vocab_story.loc[l,"word_english"]

        if letter in line:
            ind_common = ind_common.append( dict( ind = l, jp =  vocab_story.loc[l,"word_furigana"], en = en, length =  vocab_story.loc[l,"frequency"]),ignore_index=True  )

    ind_common =  ind_common.sort_values("length", ascending = False)
    ind_story =  ind_story.sort_values("length", ascending = False)

    ind_common = ind_common.drop_duplicates('jp')
    ind_story = ind_story.drop_duplicates('jp')
    ind_common = ind_common.sort_values("length", ascending = False).reset_index()
    ind_story = ind_story.sort_values("length", ascending = False).reset_index()


    if len(ind_common) > 1:
        #get shortest examples
        if i == 0:
            ind = [0,1]
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
