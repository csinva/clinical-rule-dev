import pathlib
import re
from typing import Dict, List
import numpy as np


from collections import defaultdict
import pandas as pd
from os.path import join
import os.path
from tqdm import tqdm
import json
import os
import numpy as np
import openai
from os.path import dirname
from paper_setup import papers_dir
import imodelsx


def get_paper_text(id, papers_dir=papers_dir):
    paper_file = join(papers_dir, str(int(id)) + ".txt")
    real_input = pathlib.Path(paper_file).read_text()
    return real_input


def clean_paper_text(s, unique_world_threshold=50):
    STRS_TO_REMOVE = [
        "Downloaded from http://journals.lww.com/amjclinicaloncology by BhDMf5ePHKav1zEoum1tQfN4a+kJLhEZgbsIH\no4XMi0hCywCX1AWnYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC1y0abggQZXdtwnfKZBYtws= on 08/09/2023\n",
        "Downloaded from http://pubs.asahq.org/anesthesiology/article-pdf/58/3/277/306763/0000542-198303000-00016.pdf by guest on 22 June 2023",
        "Downloaded from http://journals.lww.com/jtrauma by BhDMf5ePHKav1zEoum1tQfN4a+kJLhEZgbsIHo4XMi0hCywCX",
        "1AWnYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC1y0abggQZXdtwnfKZBYtws= on 07/20/2023",
        "1AWnYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC1y0abggQZXdtwnfKZBYtws= on 07/24/2023",
        "1AWnYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC1y0abggQZXdgGj2MwlZLeI= on 07/20/2023",
        "1AWnYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC4/OAVpDDa8K2+Ya6H515kE= on 07/24/2023",
        "Downloaded from http://journals.lww.com/jpgn by BhDMf5ePHKav1zEoum1tQfN4a+kJLhEZgbsIHo4XMi0hCywCX1AW",
        "nYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC1y0abggQZXdgGj2MwlZLeI= on 07/20/2023",
        "The user has requested enhancement of the downloaded file.",
        "View publication stats",
        "Downloaded from http://ahajournals.org by on June 21, 2023",
        "All rights reserved.",
        "For personal use only.",
        "Terms and Conditions",
        "for rules of use;",
        "Univ.of California Berkeley user",
        "University of California - Berkeley User",
        "Univ.of California Berkeley user",
        "University of California, Davis - Library user",
        "by University Of California - Davis",
        "copyright",
        "Academies Press",
        "Lacy et al\nGastroenterology Vol. 150, No. 6",
        "LLC",
        "Author manuscript",
        "available in PMC",
        "Univ of Calif Lib",
        "Not to be reproduced without permission",
        "Unauthorized reproduction of this article is prohibited.",
        "Copyright",
        "Full Terms & Conditions of access and use can be found at",
        "©",
        "@",
        "......",
        '. . . . . . ',
        ".\n.\n",
        "\n \n",
        "For personal use.",
        "Only reproduce with permission from",
        "Downloaded from",
        "on June 21, 2023" "by guest",
        # 'Downloaded from by on June 21, 2023',
        # 'Designed and Produced by Shared Vision, Warminster\nPrinted by Press 70 Ltd, Salisbury',
        "Reproduced with permission of copyright owner.",
        "Further reproduction prohibited without permission.",
        "Protected by copyright.",
        "Downloaded by",
        "Downloaded From",
        "Downloaded from",
        "Microsoft Bing User",
        "Jed Obra",
        "University of California Berkeley user",
        "Optometry Library University of California",
        "No other uses without permission.",
        "at Berkeley (UCB)",
        "OA articles are governed by the applicable Creative Commons License",
        "Microsoft user",
        "Downloaded",
        "Wiley Online Library",
        "REPRINTED",
        "Massachusetts Medical Society.",
        "No space constraints or color gure charges",
        "Immediate publication on acceptance",
        "Inclusion in PubMed, CAS, Scopus and Google Scholar",
        "Research which is freely available for redistribution",
        "Submit your manuscript at",
        "Published online by Cambridge University Press",
        "N Engl J Med.",
        "[PubMed]",
        "[Google Scholar]",
        "BhDMf5ePHKav1zEoum1tQfN4a+kJLhEZgbsIHo4XMi0hC\nywCX1AWnYQp/IlQrHD3i3D0OdRyi7TvSFl4Cf3VC4/OAVpDDa8KKGKV0Ymy+78",
    ]
    if isinstance(s, str):
        if len(set(s.split())) <= unique_world_threshold:
            return np.nan
        frac_non_ascii = sum([ord(c) > 128 for c in s]) / len(s)
        if frac_non_ascii > 0.3:
            return np.nan

        # articles that have this failed to parse
        if "All content following this page was uploaded by" in s:
            return np.nan

        # remove urls
        s = re.sub(r"http\S+", "", s)
        s = re.sub(r"www\S+", "", s)

        # remove emails
        s = re.sub(r"\S*@\S*\s?", "", s)

        # remove fax
        s = re.sub(r"fax: \S*\s?", "", s)

        # remove non-ascii chars
        s = "".join([i if ord(i) < 128 else " " for i in s])

        for r in STRS_TO_REMOVE:
            s = s.replace(r, "")

        s = s.strip()
        return s
    else:
        return np.nan


def check_race_keywords(
    df,
    ids_with_paper,
    KEYWORDS={
        "asian",
        "caucasian",
        "african",
        "latino",
        "hispanic",
    },
):
    def _check_keywords(text):
        text = text.lower()
        for k in KEYWORDS:
            if k in text:
                return True
        return False

    df["paper_contains_race_keywords"] = np.nan

    # run loop
    for id in tqdm(ids_with_paper):
        i = df[df.id == id].index[0]
        row = df.iloc[i]
        try:
            real_input = get_paper_text(row.id, papers_dir)
            df.loc[i, "paper_contains_race_keywords"] = int(_check_keywords(real_input))
        except Exception as e:
            pass
    return df
