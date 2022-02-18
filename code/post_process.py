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


# %% dialogue 1

# =============================================================================
#
# =============================================================================

# =============================================================================
#
# =============================================================================

# =============================================================================
#
# =============================================================================
dialogue_tags = pd.read_csv("text/output/PLA_dialogue1.txt", sep="\t", header=None)
dialogue_tags.columns = ["index"] + columns_dialogue
dialogue_tags = dialogue_tags.drop("index", axis = 1)

dialogue_tags["tags"] = ""


for t in range(len(pla_story_hir)):
    print(f"\r{t+1}/{len(pla_story_hir)}; {np.round((t+1)/len(pla_story_hir)*100, 1)}", end = "\r")
    line = pla_story_hir[t]
    if "Text File :" in line:
        tag = pla_story_hir[t]
        for character in remove_characters:
            tag = tag.replace(character, "")
        tag = tag.replace("Text File : ", "")
        if "sub_" in tag:
            add = "sub"
        elif "chap_" in tag:
            add = "chapter"
        elif  "z_area" in tag:
            add = "z_area"
        else:
            add = ""

        if len(add)==0:
            tag_h = f"#PLA::dialogue::story::{tag}"
        else:
            tag_h = f"#PLA::dialogue::story::{add}::{tag}"

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

        #print(tag_h)
        #print(line_jp)
        if len( np.where(line_jp == dialogue_tags["dialogue"])[0]) > 0:
            ind = np.where(line_jp == dialogue_tags["dialogue"])[0][0]
        
            tags_dia = dialogue_tags.loc[ind, "tags"]
            if tag_h in tags_dia:
                continue
            else:
                dialogue_tags.loc[ind, "tags"] = f"{tags_dia} {tag_h}"









# %% remove weird blanks
dialogue_blanks = copy.deepcopy(dialogue_tags)

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

# %%

dia1 = pd.read_csv("text/output/PLA_dialogue1_blanks_removed.txt", sep="\t", header=None)

dia1.columns= ["index", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "tags"]

# =============================================================================
# %%

df_vocab1 = pd.DataFrame(columns = ["word", "word_furigana", "word_english", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "tags", "dialogue_length"])
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
                 tags = dia1.loc[i, "tags"],
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
             tags = dia1.loc[i, "tags"],
             dialogue_length = len(dia1.loc[i, "dialogue"]))
        df_vocab1 =  df_vocab1.append(input_entry, ignore_index=True  )

#%%
df_vocab1_index =  df_vocab1.reset_index()
df_vocab1_index_sort =  df_vocab1_index.sort_values("dialogue_length")




# %% drop duplicates and combine tags

g = df_vocab1_index_sort.groupby("word")

combine_tags = g.agg("first")
combine_tags.update(g.agg({"tags": " ".join}))
combine_tags = combine_tags.reset_index()

combine_tags_sort =  combine_tags.sort_values("index")
combine_tags_sort = combine_tags_sort.rename(columns = {'index':'index_by_dialogue'})
combine_tags_sort = combine_tags_sort.reset_index(drop=True)


combine_tags_sort.loc[ combine_tags_sort["word_english"] == "Jade", "word_english"] = "Hisui"
combine_tags_sort.loc[ combine_tags_sort["word_english"] == "American", "word_english"] = "Candy"


combine_tags_sort = combine_tags_sort.reset_index(drop=True)

counter_vocab = collections.Counter(df_vocab1_index["word"])

counter_dict = dict(counter_vocab)

combine_tags_sort["frequency"] = np.nan
for i in range(len(combine_tags_sort)):
    word = combine_tags_sort.loc[i, "word"]
    combine_tags_sort.loc[i, "frequency"] = counter_dict[word]

combine_tags_sort["frequency"] = pd.to_numeric(combine_tags_sort["frequency"], downcast='integer')

combine_tags_sort_freq =  combine_tags_sort.sort_values("frequency", ascending = False)

#%%
df_vocab1_save = combine_tags_sort_freq.reset_index(drop=True)
df_vocab1_save = df_vocab1_save.reset_index()
df_vocab1_save = df_vocab1_save.drop("dialogue_length", axis = 1)
# %%

# remove duplicacte tags

for i in range(len(df_vocab1_save)):
    tags = df_vocab1_save.loc[i,"tags"]
    tags = tags.replace("dialogue", "vocab")
    words = tags.split()
    df_vocab1_save.loc[i,"tags"] = " ".join(sorted(set(words), key=words.index))



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
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "ん")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "そう")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])

ind = np.where(df_vocab1_save["word"] == "ギリ")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "バサ")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "どう")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])

# %%
df_vocab1_save = df_vocab1_save.reset_index(drop=True)
df_vocab1_save = df_vocab1_save.drop("index", axis = 1)
df_vocab1_save = df_vocab1_save.reset_index()


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
df_vocab1_save = swap_columns(df = df_vocab1_save, c1 = 'word', c2 = 'frequency')
df_vocab1_save = swap_columns(df = df_vocab1_save, c1 = 'index_by_dialogue', c2 = 'tags')

df_vocab1_save.to_csv("text/output/PLA_vocab1.txt", sep="\t", header=None, index=None)


#%%
# =============================================================================
#
# =============================================================================

# =============================================================================
#
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


# %% Dialogue 2
dialogue_tags = pd.read_csv("text/output/PLA_dialogue2.txt", sep="\t", header=None)
dialogue_tags.columns = ["index"] + columns_dialogue
dialogue_tags = dialogue_tags.drop("index", axis = 1)

dialogue_tags["tags"] = ""


for t in range(len(pla_common_hir)):
    print(f"\r{t+1}/{len(pla_common_hir)}; {np.round((t+1)/len(pla_common_hir)*100, 1)}", end = "\r")
    line = pla_common_hir[t]
    if "Text File :" in line:
        tag = pla_common_hir[t]
        for character in remove_characters:
            tag = tag.replace(character, "")
        tag = tag.replace("Text File : ", "")
        if "sub_" in tag:
            add = "sub"
        elif "chap_" in tag:
            add = "chapter"
        elif  "z_area" in tag:
            add = "z_area"
        else:
            add = ""

        if len(add)==0:
            tag_h = f"#PLA::dialogue::common::{tag}"
        else:
            tag_h = f"#PLA::dialogue::common::{add}::{tag}"

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

        #print(tag_h)
        #print(line_jp)
        if len( np.where(line_jp == dialogue_tags["dialogue"])[0]) > 0:
            ind = np.where(line_jp == dialogue_tags["dialogue"])[0][0]
        
            tags_dia = dialogue_tags.loc[ind, "tags"]
            if tag_h in tags_dia:
                continue
            else:
                dialogue_tags.loc[ind, "tags"] = f"{tags_dia} {tag_h}"


# %%
dialogue_blanks = copy.deepcopy(dialogue_tags)

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
#%% dialogue 2 vocab


dia2 = pd.read_csv("text/output/PLA_dialogue2_blanks_removed.txt", sep="\t", header=None)

dia2.columns= ["index", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "tags"]

# =============================================================================
# %%

df_vocab1 = pd.DataFrame(columns = ["word", "word_furigana", "word_english", "dialogue", "furigana", "kana", "english", "vocab","kanji", "kanji_style", "furigana_style", "kana_style", "vocab_style", "tags", "dialogue_length"])
# Get vocab
for i in range(len(dia2)):
    print(f"\r{i+1}/{len(dia2)}; {np.round((i+1)/len(dia2)*100,1)}        ", end = "\r")

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
                 tags = dia2.loc[i, "tags"],
                 dialogue_length = len(dia2.loc[i, "dialogue"]))
            df_vocab1 =  df_vocab1.append(input_entry, ignore_index=True  )
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
             tags = dia2.loc[i, "tags"],
             dialogue_length = len(dia2.loc[i, "dialogue"]))
        df_vocab1 =  df_vocab1.append(input_entry, ignore_index=True  )

#%%
df_vocab1_index =  df_vocab1.reset_index()
df_vocab1_index_sort =  df_vocab1_index.sort_values("dialogue_length")




# %% drop duplicates and combine tags

df_vocab1_index_sort["tags"]
df_vocab1_index_sort['tags'].isnull().values.any()
df_vocab1_index_sort.fillna('', inplace=True)

g = df_vocab1_index_sort.groupby("word")

combine_tags = g.agg("first")



combine_tags.update(g.agg({"tags": " ".join}))
combine_tags = combine_tags.reset_index()

combine_tags_sort =  combine_tags.sort_values("index")
combine_tags_sort = combine_tags_sort.rename(columns = {'index':'index_by_dialogue'})
combine_tags_sort = combine_tags_sort.reset_index(drop=True)


combine_tags_sort.loc[ combine_tags_sort["word_english"] == "Jade", "word_english"] = "Hisui"
combine_tags_sort.loc[ combine_tags_sort["word_english"] == "American", "word_english"] = "Candy"


combine_tags_sort = combine_tags_sort.reset_index(drop=True)

counter_vocab = collections.Counter(df_vocab1_index["word"])

counter_dict = dict(counter_vocab)

combine_tags_sort["frequency"] = np.nan
for i in range(len(combine_tags_sort)):
    word = combine_tags_sort.loc[i, "word"]
    combine_tags_sort.loc[i, "frequency"] = counter_dict[word]

combine_tags_sort["frequency"] = pd.to_numeric(combine_tags_sort["frequency"], downcast='integer')

combine_tags_sort_freq =  combine_tags_sort.sort_values("frequency", ascending = False)

#%%
df_vocab1_save = combine_tags_sort_freq.reset_index(drop=True)
df_vocab1_save = df_vocab1_save.reset_index()
df_vocab1_save = df_vocab1_save.drop("dialogue_length", axis = 1)
# %%

# remove duplicacte tags

for i in range(len(df_vocab1_save)):
    tags = df_vocab1_save.loc[i,"tags"]
    tags = tags.replace("dialogue", "vocab")
    words = tags.split()
    df_vocab1_save.loc[i,"tags"] = " ".join(sorted(set(words), key=words.index))




# %%




ind = np.where(df_vocab1_save["word"] == "で")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "ん")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "そう")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])

ind = np.where(df_vocab1_save["word"] == "ギリ")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "バサ")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])
ind = np.where(df_vocab1_save["word"] == "どう")[0][0]
df_vocab1_save = df_vocab1_save.drop(df_vocab1_save.index[ind])

# %%
df_vocab1_save = df_vocab1_save.reset_index(drop=True)
df_vocab1_save = df_vocab1_save.drop("index", axis = 1)
df_vocab1_save = df_vocab1_save.reset_index()


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
df_vocab1_save = swap_columns(df = df_vocab1_save, c1 = 'word', c2 = 'frequency')
df_vocab1_save = swap_columns(df = df_vocab1_save, c1 = 'index_by_dialogue', c2 = 'tags')


df_vocab1_save.to_csv("text/output/PLA_vocab2.txt", sep="\t", header=None, index=None)

