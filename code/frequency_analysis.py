#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:26:50 2022

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
import matplotlib.pyplot as plt
import seaborn as sns

import json
from functools import reduce

import scipy

import random


colors  = dict(pla = "#b92c2c", sword = "#b9732c", news = "#5b99d8", wiki = "#4645c7",
               twitter = "#742cb9", ao = "#d5baef")
colors2 = dict(pla_vs_sword = "#b94f2c")
colors_mixer =  ["#b9502c",  "#816d93", "#7a3a81", "#902c81", "#ca81a1",
                 "#85888b", "#7a5a81", "#904881", "#ca9ea1",
                 "#4f6bcf", "#6a58c5", "#a4ade6",
                 "#5f37bf", "#9585dd",
                 "#ae81d9"]
colors_mixer2 =  [colors_mixer[0], colors_mixer[1], colors_mixer[5], colors_mixer[2], colors_mixer[6], colors_mixer[9], colors_mixer[3], colors_mixer[7], colors_mixer[10], colors_mixer[12], colors_mixer[4], colors_mixer[8], colors_mixer[11], colors_mixer[13], colors_mixer[14]]

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



def plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300,
              sharex=False, sharey=False, squeeze=True):
    """
    Purpose: To make a generlized plotting function.

    Parameters
    ----------
    r : TYPE, optional
        DESCRIPTION. The default is 1.
    c : TYPE, optional
        DESCRIPTION. The default is 1.
    size_length : TYPE, optional
        DESCRIPTION. The default is None.
    size_height : TYPE, optional
        DESCRIPTION. The default is None.
    dpi : TYPE, optional
        DESCRIPTION. The default is 300.
    sharex : TYPE, optional
        DESCRIPTION. The default is False.
    sharey : TYPE, optional
        DESCRIPTION. The default is False.
    squeeze : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.

    """
    if size_length is None:
        size_length = 4 * c
    if size_height is None:
        size_height = 4 * r
    fig, axes = plt.subplots(r, c, figsize=(size_length, size_height), dpi=dpi,
                             sharex=sharex, sharey=sharey, squeeze=squeeze)
    return fig, axes

# %% Getting data

text_path = "text"

path_pla_story_hir = join(text_path, "PLA_text", "story", "ja-hiragana.txt")
path_pla_common_hir = join(text_path, "PLA_text", "common", "ja-hiragana.txt")

path_sword_story_hir = join(text_path, "sword_text", "story", "ja-hiragana.txt")
path_sword_common_hir = join(text_path, "sword_text", "common", "ja-hiragana.txt")


with open(path_pla_story_hir, encoding='utf16') as f:
    pla_story_hir = f.readlines()
with open(path_pla_common_hir, encoding='utf16') as f:
    pla_common_hir = f.readlines()
with open(path_sword_story_hir, encoding='utf16') as f:
    sword_story_hir = f.readlines()
with open(path_sword_common_hir, encoding='utf16') as f:
    sword_common_hir = f.readlines()

#removing weird non-unicode characters from text
remove_characters = ["\n", "\t", "\ue30a", "\ue30b", "\ue30c", "\ue30d", "\ue30e", "\ue30f", "\ue301", "\ue302", "\ue303", "\ue304", "\ue305", "\ue306", "\ue307",
                     "\ue308", "\ue309", "\ue310", "\ue31a", "\ue31b", "\ue31c", "\ue31d", "\ue31e", "\ue311", "\ue312", "\ue313", "\ue314", "\ue315", "\ue316", "\ue317", "\ue319", "\u3000"]


# %% Getting Kanji iin PLA and Sword
"""
This block of code goes through all the text files ("story" and "common")
and extracts the kanji characters from other kana
"""


kanji_list_pla = []
kanji_list_sword = []


print("getting PLA story kanji")
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
            kanji_list_pla.append(letter)

print("getting PLA common kanji")
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
            kanji_list_pla.append(letter)

#sword
print("getting sword story kanji")
for i in range(len(sword_story_hir)):
    line = sword_story_hir[i]

    for character in remove_characters:
        line = line.replace(character, "")

    print(f"\r{i}/{len(sword_story_hir)}; {np.round(i/len(sword_story_hir)*100,1)}   ", end = "\r")
    for l in range(len(line)):
        letter = line[l]
        unic = unicodedata.name(letter)
        # print(unic)
        if "CJK" in unic:
            kanji_list_sword.append(letter)

print("getting sword common kanji")
for i in range(len(sword_common_hir)):
    line = sword_common_hir[i]

    for character in remove_characters:
        line = line.replace(character, "")

    print(f"\r{i}/{len(sword_common_hir)}; {np.round(i/len(sword_common_hir)*100,1)}   ", end = "\r")
    for l in range(len(line)):
        letter = line[l]
        unic = unicodedata.name(letter)
        # print(unic)
        if "CJK" in unic:
            kanji_list_sword.append(letter)
# %% Quantify frequency distributions
"""
This block of code takes the kanji lists from above and figures out the frequency
of all the unique kanji. It returns a dictionary of the frequency
"""

kanji_unique_pla = np.unique(kanji_list_pla)
counter_pla = collections.Counter(kanji_list_pla)
counter_dict_pla = dict(counter_pla)
df_pla = pd.DataFrame.from_dict(counter_dict_pla.items())
df_pla.columns = ["kanji", "frequency"]

kanji_unique_sword = np.unique(kanji_list_sword)
counter_sword = collections.Counter(kanji_list_sword)
counter_dict_sword = dict(counter_sword)
df_sword = pd.DataFrame.from_dict(counter_dict_sword.items())
df_sword.columns = ["kanji", "frequency"]



# %% Calculating kanji metrics

"""
This block of code quantifies the ranks, relative ranks to max
"""



df_pla["relative_to_max"] = np.nan
df_pla["relative_to_total"]  = np.nan
df_sword["relative_to_max"] = np.nan
df_sword["relative_to_total"]  = np.nan


df_pla = df_pla.sort_values(by='frequency', ascending=False)
df_pla = df_pla.reset_index(drop=True)
df_pla = df_pla.reset_index()
df_pla = swap_columns(df_pla, "index", "kanji")
df_pla = df_pla.rename(columns = {'index':'rank'})

df_sword = df_sword.sort_values(by='frequency', ascending=False)
df_sword = df_sword.reset_index(drop=True)
df_sword = df_sword.reset_index()
df_sword = swap_columns(df_sword, "index", "kanji")
df_sword = df_sword.rename(columns = {'index':'rank'})

df_pla["relative_to_total"] = df_pla["frequency"] /np.sum(df_pla["frequency"])
df_pla["relative_to_max"] = df_pla["frequency"] / np.max(df_pla["frequency"])
df_sword["relative_to_total"] = df_sword["frequency"] /np.sum(df_sword["frequency"])
df_sword["relative_to_max"] = df_sword["frequency"] / np.max(df_sword["frequency"])


df_pkmn = df_pla.merge(df_sword, left_on='kanji', right_on='kanji', suffixes = ["_pla", "_sword"], how = "outer")


df_pkmn['frequency_pla'] = df_pkmn['frequency_pla'].fillna(0)
df_pkmn['relative_to_max_pla'] = df_pkmn['relative_to_max_pla'].fillna(0)
df_pkmn['relative_to_total_pla'] = df_pkmn['relative_to_total_pla'].fillna(0)
df_pkmn['rank_pla'] = df_pkmn['rank_pla'].fillna(  np.max(df_pkmn['rank_pla'])+1  )

df_pkmn['frequency_sword'] = df_pkmn['frequency_sword'].fillna(0)
df_pkmn['relative_to_max_sword'] = df_pkmn['relative_to_max_sword'].fillna(0)
df_pkmn['relative_to_total_sword'] = df_pkmn['relative_to_total_sword'].fillna(0)
df_pkmn['rank_sword'] = df_pkmn['rank_sword'].fillna(  np.max(df_pkmn['rank_sword'])+1  )

df_pkmn_inner = df_pla.merge(df_sword, left_on='kanji', right_on='kanji', suffixes = ["_pla", "_sword"])

# %%
"""
sns.scatterplot( data = df_pkmn, x = "frequency_pla", y = "frequency_sword")
sns.scatterplot( data = df_pkmn, x = "rank_pla", y = "rank_sword")
scipy.stats.spearmanr(df_pkmn["rank_pla"], df_pkmn["rank_sword"])[0]
"""


# %%
#https://scriptin.github.io/kanji-frequency/

path_aozora = join(text_path, "public_data", "aozora.json")
path_news = join(text_path, "public_data", "news.json")
path_twitter = join(text_path, "public_data", "twitter.json")
path_wikipedia = join(text_path, "public_data", "wikipedia.json")

columns = ["kanji", "frequency", "relative_to_total"]

with open(path_aozora, 'r') as f:
  public_aozora = json.load(f)
with open(path_news, 'r') as f:
  public_news = json.load(f)
with open(path_twitter, 'r') as f:
  public_twitter = json.load(f)
with open(path_wikipedia, 'r') as f:
  public_wikipedia  = json.load(f)


public_aozora = pd.DataFrame.from_records(public_aozora, columns = columns)
public_news = pd.DataFrame.from_records(public_news, columns = columns)
public_twitter = pd.DataFrame.from_records(public_twitter, columns = columns)
public_wikipedia = pd.DataFrame.from_records(public_wikipedia, columns = columns)

public_aozora = public_aozora.drop(0)
public_news = public_news.drop(0)
public_twitter = public_twitter.drop(0)
public_wikipedia = public_wikipedia.drop(0)


public_aozora = public_aozora.reset_index(drop=True)
public_aozora = public_aozora.reset_index()
public_aozora = swap_columns(public_aozora, "index", "kanji")
public_aozora = public_aozora.rename(columns = {'index':'rank'})

public_news = public_news.reset_index(drop=True)
public_news = public_news.reset_index()
public_news = swap_columns(public_news, "index", "kanji")
public_news = public_news.rename(columns = {'index':'rank'})

public_twitter = public_twitter.reset_index(drop=True)
public_twitter = public_twitter.reset_index()
public_twitter = swap_columns(public_twitter, "index", "kanji")
public_twitter = public_twitter.rename(columns = {'index':'rank'})

public_wikipedia = public_wikipedia.reset_index(drop=True)
public_wikipedia = public_wikipedia.reset_index()
public_wikipedia = swap_columns(public_wikipedia, "index", "kanji")
public_wikipedia = public_wikipedia.rename(columns = {'index':'rank'})

public_aozora["relative_to_max"] = np.nan
public_news["relative_to_max"] = np.nan
public_twitter["relative_to_max"] = np.nan
public_wikipedia["relative_to_max"] = np.nan

public_aozora["relative_to_max"] = public_aozora["frequency"] / np.max(public_aozora["frequency"])
public_news["relative_to_max"] = public_news["frequency"] / np.max(public_news["frequency"])
public_twitter["relative_to_max"] = public_twitter["frequency"] / np.max(public_twitter["frequency"])
public_wikipedia["relative_to_max"] = public_wikipedia["frequency"] / np.max(public_wikipedia["frequency"])

# %% graph of frequuency distrbutions




def plot_zipf(public_news, slope = -8, intercept = 23, start = 3, stop = None,
              color_data = "#6666aa" , color_line1 = "#111111bb", color_line2 = "#33333399",
              ls1 = "-", ls2 = "--",size = 30, xlim = None , lw = 3):
    
    fig, axes = plot_make(size_length = 10)
    x1 = np.array(range(len(public_news)))+1
    y1 = 1/x1
    y2 = 1/(x1**0.5)
    #y3 = 1/(x1**0.25)
    y4 = (1/(  (x1)**.25))

    y4 = 10**(slope*np.log10(x1)+ intercept)


    sns.lineplot(x =  x1, y =  y2, ax = axes , color =color_line1 , ls = ls1, lw = lw)
    if stop == None:
        sns.lineplot(x =  x1[int(10**start):], y =  y4[int(10**start):], ax = axes , color = color_line2 , ls = ls2 , lw = lw)
    else:
        sns.lineplot(x =  x1[int(10**start):-int(10**stop)], y =  y4[int(10**start):-int(10**stop)], ax = axes , color = color_line2 , ls = ls2, lw = lw )

    #sns.lineplot(x =  x1, y =  y1, ax = axes , color = "red" )
    #sns.lineplot(x =  x1, y =  y3, ax = axes , color = "purple" )

    sns.scatterplot(data = public_news, x = "rank", y = "relative_to_max", ax = axes ,edgecolor="none", s = size, color = color_data )


    axes.set_yscale("log")
    axes.set_xscale("log")
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    if not xlim == None:
        axes.set_xlim(xlim)

xlim = [-1,25000]
#%%
plot_zipf(df_pla, slope = -3, intercept = 6.5, start = 2.5, xlim = xlim , color_data = colors["pla"])
plot_zipf(df_sword, slope = -5, intercept = 12.5, start = 2.5, xlim =xlim, color_data = colors["sword"] )


plot_zipf(public_news, slope = -8, intercept = 23, xlim = xlim, color_data = colors["news"])
plot_zipf(public_wikipedia, slope = -6, intercept = 17, stop = 4,  xlim = xlim, color_data = colors["wiki"])
plot_zipf(public_twitter, slope = -6, intercept = 16.5, start = 2.8,  xlim =xlim , color_data = colors["twitter"])
plot_zipf(public_aozora, slope = -10, intercept = 32, start = 3.3,  xlim = xlim , color_data = colors["ao"])



# %%

#Figure 3
fig, axes = plot_make()
sns.scatterplot( data = df_pkmn_inner, x = "frequency_pla", y = "frequency_sword", ax = axes, edgecolor = None, color = colors2["pla_vs_sword"], s = 5)

sns.lineplot(x = [0,1600], y = [0,1600],  ax = axes, color = "#33333399", ls = "--", lw = 4)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_yscale("log")
axes.set_xscale("log")

#%%
fig, axes = plot_make()
sns.scatterplot( data = df_pkmn_inner, x = "rank_pla", y = "rank_sword", ax = axes,edgecolor = None, color = colors2["pla_vs_sword"], s = 5)
scipy.stats.spearmanr(df_pkmn_inner["rank_pla"], df_pkmn_inner["rank_sword"])[0]

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)



fig, axes = plot_make()
sns.scatterplot( data = df_pkmn_inner, x = "rank_pla", y = "rank_sword", ax = axes,edgecolor = None, color = colors2["pla_vs_sword"], s = 5)

sns.lineplot(x = [0,1600], y = [0,1600],  ax = axes, color = "#33333399", ls = "--", lw = 4)

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)



# %%

cutoff = np.max([np.min(df_pla["relative_to_total"]), np.min(df_sword["relative_to_total"])]) #making kanji cutoff to the lowest

np.min(public_aozora["relative_to_total"])
np.min(public_news["relative_to_total"])
np.min(public_twitter["relative_to_total"])
np.min(public_wikipedia["relative_to_total"])


public_aozora_cutoff = public_aozora[public_aozora["relative_to_total"] >=cutoff]
public_news_cutoff = public_news[public_news["relative_to_total"] >=cutoff]
public_twitter_cutoff = public_twitter[public_twitter["relative_to_total"] >=cutoff]
public_wikipedia_cutoff = public_wikipedia[public_wikipedia["relative_to_total"] >=cutoff]

df_pla_cutoff = df_pla[df_pla["relative_to_total"] >=cutoff]
df_sword_cutoff = df_sword[df_sword["relative_to_total"] >=cutoff]


# %%
dfs = [df_pla_cutoff, df_sword_cutoff , public_news_cutoff, public_wikipedia_cutoff, public_twitter_cutoff, public_aozora_cutoff]

names = ["pla", "sword", "news", "wikipedia", "twitter", "aozora"]
for i in range(len(dfs)):
    dfs[i] = dfs[i].rename(columns = {'rank':f'rank_{names[i]}'})
    dfs[i] = dfs[i].rename(columns = {'frequency':f'frequency_{names[i]}'})
    dfs[i] = dfs[i].rename(columns = {'relative_to_max':f'relative_to_max_{names[i]}'})
    dfs[i] = dfs[i].rename(columns = {'relative_to_total':f'relative_to_total_{names[i]}'})


df_merged_outer = reduce(lambda left,right: pd.merge(left,right,on='kanji', how = "outer"), dfs)
df_merged= reduce(lambda left,right: pd.merge(left,right,on='kanji'), dfs)


df_merged_outer['frequency_pla'] = df_merged_outer['frequency_pla'].fillna(0)
df_merged_outer['relative_to_max_pla'] = df_merged_outer['relative_to_max_pla'].fillna(0)
df_merged_outer['relative_to_total_pla'] = df_merged_outer['relative_to_total_pla'].fillna(0)
df_merged_outer['rank_pla'] = df_merged_outer['rank_pla'].fillna(  np.max(df_merged_outer['rank_pla'])+1  )

df_merged_outer['frequency_sword'] = df_merged_outer['frequency_sword'].fillna(0)
df_merged_outer['relative_to_max_sword'] = df_merged_outer['relative_to_max_sword'].fillna(0)
df_merged_outer['relative_to_total_sword'] = df_merged_outer['relative_to_total_sword'].fillna(0)
df_merged_outer['rank_sword'] = df_merged_outer['rank_sword'].fillna(  np.max(df_merged_outer['rank_sword'])+1  )

df_merged_outer['frequency_aozora'] = df_merged_outer['frequency_aozora'].fillna(0)
df_merged_outer['relative_to_max_aozora'] = df_merged_outer['relative_to_max_aozora'].fillna(0)
df_merged_outer['relative_to_total_aozora'] = df_merged_outer['relative_to_total_aozora'].fillna(0)
df_merged_outer['rank_aozora'] = df_merged_outer['rank_aozora'].fillna(  np.max(df_merged_outer['rank_aozora'])+1  )

df_merged_outer['frequency_news'] = df_merged_outer['frequency_news'].fillna(0)
df_merged_outer['relative_to_max_news'] = df_merged_outer['relative_to_max_news'].fillna(0)
df_merged_outer['relative_to_total_news'] = df_merged_outer['relative_to_total_news'].fillna(0)
df_merged_outer['rank_news'] = df_merged_outer['rank_news'].fillna(  np.max(df_merged_outer['rank_news'])+1  )

df_merged_outer['frequency_twitter'] = df_merged_outer['frequency_twitter'].fillna(0)
df_merged_outer['relative_to_max_twitter'] = df_merged_outer['relative_to_max_twitter'].fillna(0)
df_merged_outer['relative_to_total_twitter'] = df_merged_outer['relative_to_total_twitter'].fillna(0)
df_merged_outer['rank_twitter'] = df_merged_outer['rank_twitter'].fillna(  np.max(df_merged_outer['rank_twitter'])+1  )

df_merged_outer['frequency_wikipedia'] = df_merged_outer['frequency_wikipedia'].fillna(0)
df_merged_outer['relative_to_max_wikipedia'] = df_merged_outer['relative_to_max_wikipedia'].fillna(0)
df_merged_outer['relative_to_total_wikipedia'] = df_merged_outer['relative_to_total_wikipedia'].fillna(0)
df_merged_outer['rank_wikipedia'] = df_merged_outer['rank_wikipedia'].fillna(  np.max(df_merged_outer['rank_wikipedia'])+1  )

np.max(df_merged["rank_sword"])
np.max(df_merged["rank_pla"])



scipy.stats.spearmanr(df_merged["rank_pla"],df_merged["rank_sword"])[0]
scipy.stats.spearmanr(df_merged["rank_pla"],df_merged["rank_aozora"])[0]
scipy.stats.spearmanr(df_merged["rank_wikipedia"],df_merged["rank_news"])[0]

df_ranks = df_merged[['rank_pla','rank_sword','rank_news', 'rank_wikipedia', 'rank_twitter', "rank_aozora"]]
#%%
#Fig 3 summary
i,j = 0,1
fig, axes = plot_make()
c1 = df_ranks.columns[i]
c2 = df_ranks.columns[j]

sns.lineplot(x = [0,1600], y = [0,1600],  ax = axes, color = "#33333399", ls = "--", lw = 6)
sns.regplot( data = df_merged, x = c1, ax = axes, y = c2,scatter_kws=dict( edgecolor = None, s = 5), color =  colors_mixer[0], ci = None, line_kws=dict(lw=6))


axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)

corr = scipy.stats.spearmanr(df_merged[c1], df_merged[c2])[0]
axes.text(s = f"r={(np.round(corr,2))}", x = 0, y = 1600, size = 15, va = "top")

plt.savefig("plots/fig3_blowup.pdf", dpi=600)

#%%

#Fig 3 all corrletions

fig, axes = plot_make(r = 6, c = 6)
df_ranks.columns
count1 = 0
count2 = 0
colors_mixer

for i in range(6):
    for j in range(6):
        if i == j or i>j:
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['left'].set_visible(False)
            axes[i][j].spines['bottom'].set_visible(False)
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].get_yaxis().set_visible(False)
            continue

        c1 = df_ranks.columns[i]
        c2 = df_ranks.columns[j]

        if i <j:
            sns.regplot( data = df_merged, x = c1, y = c2, ax = axes[i][j], scatter_kws=dict( edgecolor = None, s = 5), color =  colors_mixer[count1], ci = None, line_kws=dict(lw=6))
            sns.lineplot(x = [0,1600], y = [0,1600],  ax = axes[i][j], color = "#33333399", ls = "--", lw = 6)
            count1 = count1+1
            max_y = np.max(df_merged[c2])
            corr = scipy.stats.spearmanr(df_merged[c1], df_merged[c2])[0]
            axes[i][j].text(s = f"r={(np.round(corr,2))}", x = 0, y = max_y, size = 20, va = "top")
        """
        if i>j:

            change_rank =np.abs(df_merged[c1] -  df_merged[c2])
            mean_change_rank = np.nanmedian(change_rank)
            sns.histplot(change_rank, ax = axes[i][j], kde = True,edgecolor=None, color = colors_mixer2[count2])
            print(f"{c1}, {c2}, {np.round(mean_change_rank,1)}")
            count2 = count2+1
            axes[i][j].set_xlim([-5,2100])
            axes[i][j].set_ylim([0,290])
        """
        axes[i][j].spines['right'].set_visible(False)
        axes[i][j].spines['top'].set_visible(False)
        axes[i][j].set_xlabel("")
        axes[i][j].set_ylabel("")

plt.savefig("plots/fig3_allcorr.pdf", dpi=600)
# %%
#Find the largest change in ranks
changes = pd.DataFrame(columns = ["first", "second", "kanji", "rank1", "rank2", "change"])
for i in range(6):
    for j in range(6):
        if i == j or i>j:
            continue
        c1 = df_ranks.columns[i]
        c2 = df_ranks.columns[j]
        change_rank =np.abs(df_merged[c1] -  df_merged[c2])
        mean_change_rank = np.nanmedian(change_rank)

        largest_change = -np.sort(-change_rank )[0:5]
        for c in range(len(largest_change)):
            ind = np.where(change_rank == largest_change[c])[0][0]
            df_merged.loc[ind]
            which_one = df_merged[ ["kanji", c1, c2] ].loc[ind]
            changes = changes.append(dict( first = c1 , second = c2, kanji =
                                          which_one["kanji"], rank1 = which_one[c1],
                                          rank2 = which_one[c2],
                                          change = which_one[c2] -  which_one[c1]),
                                     ignore_index=True)


        np.abs(df_merged[c1] -  df_merged[c2])
        print(f"{c1}, {c2}, {np.round(mean_change_rank,1)}")

        mean_change_rank = np.nanmedian(np.abs(df_merged[c1] -  df_merged[c2]))





# %%
i=0
j=1

c1 = df_ranks.columns[i]
c2 = df_ranks.columns[j]

print(f"{c1}, {c2}")
df_merged[c1]
df_merged[c2]

fig, axes = plot_make(r = 1, c = 1)
total = 100
samples = random.choices(range(len(df_merged)) , k= total)

for s in range(len(samples)):
    p = samples[s]
    y1 = df_merged[c1][p]+1
    y2 =  df_merged[c2][p]+1
    if y1 > y2+100:
        col = "#aa333322" #red
    elif y2 > y1+100:
        col = "#3333aa22" #blue
    else:
        col = "#33333322"
    print(f"\r{s}/{total}, {np.round((s+1)/total*100, 2)}  ", end = "\r")
    sns.lineplot(  x = [0,1], y =[y1,y2  ], ax = axes,lw = 1  , color = col )

axes.set_ylim(axes.get_ylim()[::-1])





# %%

fig, axes = plot_make(r = 6, c = 6)
count1 = 0
count2 = 0
colors_mixer

for i in range(6):
    for j in range(6):
        if i == j:
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['left'].set_visible(False)
            axes[i][j].spines['bottom'].set_visible(False)
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].get_yaxis().set_visible(False)
            continue

        c1 = df_ranks.columns[i]
        c2 = df_ranks.columns[j]
        if i<j:

            change_rank =np.abs((df_merged[c1] -  df_merged[c2]))
            mean_change_rank = np.nanmedian(np.abs(change_rank))
            sns.histplot(change_rank, ax = axes[i][j], kde = True,edgecolor=None, color = colors_mixer[count2])
            #axes[i][j].axvline(mean_change_rank, ls = "--", color = "#000000")
            axes[i][j].text(s = f"{int(np.round(mean_change_rank,0))}", x = 900, y = 150, size = 30)
            print(f"{c1}, {c2}, {np.round(mean_change_rank,1)}")
            count2 = count2+1
            axes[i][j].set_xlim([-5,2100])
            axes[i][j].set_ylim([0,290])

        if i >j:
            total = 100
            samples = random.choices(range(len(df_merged)) , k= total)
            for s in range(len(samples)):
                p = samples[s]
                y1 = df_merged[c1][p]+1
                y2 =  df_merged[c2][p]+1
                switch_rank_cutoff = 300
                if y1 > y2+switch_rank_cutoff:
                    col = "#aa333333" #red
                elif y2 > y1+switch_rank_cutoff:
                    col = "#3333aa33" #blue
                else:
                    col = "#33333322"
                print(f"\r{s}/{total}, {np.round((s+1)/total*100, 2)}  ", end = "\r")
                sns.lineplot(  x = [0,1], y =[y1,y2  ], ax = axes[i][j],lw = 2, color = col )

            #axes[i][j].set_xlim([-3,2100])
            axes[i][j].set_ylim(axes[i][j].get_ylim()[::-1])

        axes[i][j].spines['right'].set_visible(False)
        axes[i][j].spines['top'].set_visible(False)
        axes[i][j].set_xlabel("")
        axes[i][j].set_ylabel("")









# %%


df_ranks_outer = df_merged_outer[['rank_pla','rank_sword','rank_news', 'rank_wikipedia', 'rank_twitter', 'rank_aozora']]



#sns.pairplot(df_ranks_outer)

scipy.stats.spearmanr(df_ranks["rank_pla"], df_ranks["rank_sword"])[0]
scipy.stats.spearmanr(df_ranks["rank_pla"],df_ranks["rank_news"])[0]
scipy.stats.spearmanr(df_ranks["rank_pla"],df_ranks["rank_wikipedia"])[0]
scipy.stats.spearmanr(df_ranks["rank_pla"],df_ranks["rank_twitter"])[0]
scipy.stats.spearmanr(df_ranks["rank_pla"],df_ranks["rank_aozora"])[0]

scipy.stats.spearmanr(df_ranks["rank_sword"],df_ranks["rank_news"])[0]
scipy.stats.spearmanr(df_ranks["rank_sword"],df_ranks["rank_wikipedia"])[0]
scipy.stats.spearmanr(df_ranks["rank_sword"],df_ranks["rank_twitter"])[0]
scipy.stats.spearmanr(df_ranks["rank_sword"],df_ranks["rank_aozora"])[0]

scipy.stats.spearmanr(df_ranks["rank_news"],df_ranks["rank_wikipedia"])[0]
scipy.stats.spearmanr(df_ranks["rank_news"],df_ranks["rank_twitter"])[0]
scipy.stats.spearmanr(df_ranks["rank_news"],df_ranks["rank_aozora"])[0]

scipy.stats.spearmanr(df_ranks["rank_twitter"],df_ranks["rank_wikipedia"])[0]
scipy.stats.spearmanr(df_ranks["rank_twitter"],df_ranks["rank_aozora"])[0]

scipy.stats.spearmanr(df_ranks["rank_wikipedia"],df_ranks["rank_aozora"])[0]




# %%
# Bootstrap


n = len(df_ranks)
df_merged
# %%
#hypothesis: pla and sword more correlated than pla and news

#HYPOTHESIS: to learn pokemon, one shoould focus on outside text such as books and real-world
#scnarios such as twitter than more formal resources such as news and wikipedia that function to
#disseminate information, rather than dialogue and expressions.

#sub-hypothesis: pokemon are more correlated to twitter/books than to new or wiki

#cconclusion: immesion in twitter/books may be moderately useful for helping learn japanese
#in the intended media - lke pokemon. However, nothing subsititues for studying direcctly
#to the vocabulary and language for which you intend to use it for. Hencce why we
#provide the study resources here - and the importance of using directed study materials
#wikpediia seems the most different in kanji use from pokemon,
#understanding wikipedia may be less useful as a marker for reading pokemon.
#Conversely understanding pokemon may be less useful as a marker for reading wikipedia

B = 10000
cor1 = np.zeros( shape = (B))
cor2 = np.zeros( shape = (B))

for i in range(B):
    print(f"\r{i+1}/{B}; {np.round((i+1)/B*100,1)}", end = "\r")
    samples = random.choices(range(n) , k= n)
    np.array(samples)
    df_sample = df_ranks.iloc[samples]

    cor1[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_news"])[0]
    cor2[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_twitter"])[0]

#%%
fig, axes = plot_make(r=1, c=1, size_length=3, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor1, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[1], binrange = [0.4, 0.6], binwidth = 0.01 , line_kws = dict(lw = 3))
sns.histplot(cor2, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[3], binrange = [0.4, 0.6], binwidth = 0.01, line_kws = dict(lw = 3))
axes.axvline(np.mean(cor1), color= colors_mixer[1], ls = "--")
axes.axvline(np.mean(cor2), color= colors_mixer[3], ls = "--")

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
pval = len(np.where(cor2-cor1 < 0)[0])/B
print(f"pvalue: {pval}")
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
axes.text(s = f"p={(np.round(pval,3))}", x = xmin + (xmax-xmin)*0.05, y = ymax - (ymax-ymin)*0.05, size = 10, va = "top")

plt.savefig("plots/fig3_planews_vs_platwitter.pdf", dpi=600)
#fig, axes = plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
#sns.histplot(cor2-cor1, ax = axes)
#%%

B = 10000
cor1 = np.zeros( shape = (B))
cor2 = np.zeros( shape = (B))

for i in range(B):
    print(f"\r{i+1}/{B}; {np.round((i+1)/B*100,1)}", end = "\r")
    samples = random.choices(range(n) , k= n)
    np.array(samples)
    df_sample = df_ranks.iloc[samples]

    cor1[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_wikipedia"])[0]
    cor2[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_twitter"])[0]

#%%
fig, axes = plot_make(r=1, c=1, size_length=3, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor1, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[2], binrange = [0.4, 0.6], binwidth = 0.01 , line_kws = dict(lw = 3))
sns.histplot(cor2, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[3], binrange = [0.4, 0.6], binwidth = 0.01, line_kws = dict(lw = 3))
axes.axvline(np.mean(cor1), color= colors_mixer[2], ls = "--")
axes.axvline(np.mean(cor2), color= colors_mixer[3], ls = "--")

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
pval = len(np.where(cor2-cor1 < 0)[0])/B
print(f"pvalue: {pval}")
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
axes.text(s = f"p={(np.round(pval,3))}", x = xmin + (xmax-xmin)*0.05, y = ymax - (ymax-ymin)*0.05, size = 10, va = "top")

plt.savefig("plots/fig3_plawiki_vs_platwitter.pdf", dpi=600)

#%%

B = 10000
cor1 = np.zeros( shape = (B))
cor2 = np.zeros( shape = (B))

for i in range(B):
    print(f"\r{i+1}/{B}; {np.round((i+1)/B*100,1)}", end = "\r")
    samples = random.choices(range(n) , k= n)
    np.array(samples)
    df_sample = df_ranks.iloc[samples]

    cor1[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_news"])[0]
    cor2[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_aozora"])[0]


#%%
fig, axes = plot_make(r=1, c=1, size_length=3, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor1, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[1], binrange = [0.4, 0.6], binwidth = 0.01 , line_kws = dict(lw = 3))
sns.histplot(cor2, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[4], binrange = [0.4, 0.6], binwidth = 0.01, line_kws = dict(lw = 3))
axes.axvline(np.mean(cor1), color= colors_mixer[1], ls = "--")
axes.axvline(np.mean(cor2), color= colors_mixer[4], ls = "--")

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
pval = len(np.where(cor2-cor1 < 0)[0])/B
print(f"pvalue: {pval}")
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
axes.text(s = f"p={(np.round(pval,3))}", x = xmin + (xmax-xmin)*0.05, y = ymax - (ymax-ymin)*0.05, size = 10, va = "top")

plt.savefig("plots/fig3_planews_vs_plaao.pdf", dpi=600)


# %%

B = 10000
cor1 = np.zeros( shape = (B))
cor2 = np.zeros( shape = (B))

for i in range(B):
    print(f"\r{i+1}/{B}; {np.round((i+1)/B*100,1)}", end = "\r")
    samples = random.choices(range(n) , k= n)
    np.array(samples)
    df_sample = df_ranks.iloc[samples]

    cor1[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_wikipedia"])[0]
    cor2[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_aozora"])[0]

#%%
fig, axes = plot_make(r=1, c=1, size_length=3, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor1, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[2], binrange = [0.4, 0.6], binwidth = 0.01 , line_kws = dict(lw = 3))
sns.histplot(cor2, ax = axes,  kde = True, edgecolor=None, color = colors_mixer[4], binrange = [0.4, 0.6], binwidth = 0.01, line_kws = dict(lw = 3))
axes.axvline(np.mean(cor1), color= colors_mixer[2], ls = "--")
axes.axvline(np.mean(cor2), color= colors_mixer[4], ls = "--")

axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
pval = len(np.where(cor2-cor1 < 0)[0])/B
print(f"pvalue: {pval}")
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
axes.text(s = f"p={(np.round(pval,3))}", x = xmin + (xmax-xmin)*0.05, y = ymax - (ymax-ymin)*0.05, size = 10, va = "top")

plt.savefig("plots/fig3_plawiki_vs_plaao.pdf", dpi=600)













# %%


# %%
B = 10000
cor1 = np.zeros( shape = (B))
cor2 = np.zeros( shape = (B))



for i in range(B):
    print(f"\r{i+1}/{B}; {np.round((i+1)/B*100,1)}   ", end = "\r")
    samples = random.choices(range(n) , k= n)
    np.array(samples)
    df_sample = df_ranks.iloc[samples]

    cor1[i] = scipy.stats.spearmanr(df_sample["rank_pla"], df_sample["rank_sword"])[0]
    cor2[i] = scipy.stats.spearmanr(df_sample["rank_news"], df_sample["rank_wikipedia"])[0]


fig, axes = plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor1, ax = axes)
sns.histplot(cor2, ax = axes)

fig, axes = plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor2-cor1, ax = axes)

len(np.where(  cor1 > np.mean(cor2)   )[0])/B


pval = len(np.where(cor2-cor1 < 0)[0])/B
print(f"pvalue: {pval}")



# %%







fig, axes = plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.histplot(cor2-cor1, ax = axes)



# %% Franchises

sales = pd.DataFrame(columns = ["franchise", "year", "revenue"])


sales = sales.append(dict(franchise = "pokemon", year = 1996, revenue = 109), ignore_index=True)
sales = sales.append(dict(franchise = "hello_kitty", year = 1974, revenue = 88.5), ignore_index=True)
sales = sales.append(dict(franchise = "michey_mouse_and_friends", year = 1928, revenue = 82.9), ignore_index=True)
sales = sales.append(dict(franchise = "winnie_the_pooh", year = 1924, revenue = 81), ignore_index=True)
sales = sales.append(dict(franchise = "star_wars", year = 1977, revenue = 69.4), ignore_index=True)
sales = sales.append(dict(franchise = "mario", year = 1981, revenue = 55.1), ignore_index=True)
sales = sales.append(dict(franchise = "dinsey_princess", year = 2000, revenue = 46.4), ignore_index=True)
sales = sales.append(dict(franchise = "anpanman", year = 1973, revenue = 44.9), ignore_index=True)
sales = sales.append(dict(franchise = "MCU", year = 2008, revenue = 38), ignore_index=True)
sales = sales.append(dict(franchise = "harry_potter", year = 1997, revenue = 32.2), ignore_index=True)

sns.barplot(x="franchise", y="revenue", data=sales, color="salmon", saturation=.5)
# %% GDP

gdp = pd.DataFrame(columns = ["country", "gdp"])


gdp = gdp.append(dict(country = "US", gdp = 22.94), ignore_index=True)
gdp = gdp.append(dict(country = "China", gdp = 16.86), ignore_index=True)
gdp = gdp.append(dict(country = "Japan", gdp = 5.10), ignore_index=True)
gdp = gdp.append(dict(country = "Germany", gdp = 4.23), ignore_index=True)
gdp = gdp.append(dict(country = "UK", gdp = 3.11), ignore_index=True)
gdp = gdp.append(dict(country = "India", gdp = 2.95), ignore_index=True)
gdp = gdp.append(dict(country = "France",  gdp = 2.94), ignore_index=True)
gdp = gdp.append(dict(country = "Italy", gdp = 2.12), ignore_index=True)
gdp = gdp.append(dict(country = "Canada", gdp = 2.02), ignore_index=True)
gdp = gdp.append(dict(country = "Korea",  gdp = 1.82), ignore_index=True)

palette=["#777777", "#777777","#ff1111","#777777","#777777","#777777","#777777","#777777","#777777","#777777"]

fig, axes = plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300, sharex=False, sharey=False, squeeze=True)
sns.barplot(x="country", y="gdp", data=gdp, palette=palette, saturation=.5,alpha = 0.9, ax = axes)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(6)
axes.spines['left'].set_linewidth(6)
axes.tick_params(width=6, length=8)

plt.savefig("plots/gdp.pdf", dpi=600)

# %%kanji in pla not in sword



df_overlap = df_pla.merge(df_sword, left_on='kanji', right_on='kanji', suffixes = ["_pla", "_sword"], how = "outer")

np.array(df_overlap[df_overlap["rank_pla"].isnull()]["kanji"])
np.array(df_overlap[df_overlap["rank_sword"].isnull()]["kanji"])


df_overlap = df_pla.merge(public_news, left_on='kanji', right_on='kanji', suffixes = ["_pla", "_news"], how = "outer")
np.array(df_overlap[df_overlap["rank_pla"].isnull()]["kanji"])
np.array(df_overlap[df_overlap["rank_news"].isnull()]["kanji"])


df_overlap = df_pla.merge(public_news_cutoff, left_on='kanji', right_on='kanji', suffixes = ["_pla", "_news"], how = "outer")
len(np.array(df_overlap[df_overlap["rank_pla"].isnull()]["kanji"]))
len(np.array(df_overlap[df_overlap["rank_news"].isnull()]["kanji"]))





cutoff = np.max([np.min(df_pla["relative_to_total"]), np.min(df_sword["relative_to_total"])]) #making kanji cutoff to the lowest

np.min(public_aozora["relative_to_total"])
np.min(public_news["relative_to_total"])
np.min(public_twitter["relative_to_total"])
np.min(public_wikipedia["relative_to_total"])


public_aozora_cutoff = public_aozora[public_aozora["relative_to_total"] >cutoff]
public_news_cutoff = public_news[public_news["relative_to_total"] >cutoff]
public_twitter_cutoff = public_twitter[public_twitter["relative_to_total"] >cutoff]
public_wikipedia_cutoff = public_wikipedia[public_wikipedia["relative_to_total"] >cutoff]

df_pla_cutoff = df_pla[df_pla["relative_to_total"] >cutoff]
df_sword_cutoff = df_sword[df_sword["relative_to_total"] >cutoff]