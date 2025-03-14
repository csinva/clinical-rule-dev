import pathlib
import re
from typing import Dict, List
import numpy as np


from collections import defaultdict
import fitz
import pandas as pd
from os.path import join
import os.path
from tqdm import tqdm
import json
import os
import numpy as np
import pubmed
import openai
from os.path import dirname

path_to_file = dirname(__file__)
path_to_repo = dirname(path_to_file)
papers_dir = join(path_to_repo, "papers")


# def download_open_source_papers(df: pd.DataFrame):
# # download papers
# refs = get_updated_refs(df)
# all_ids = df.id
# ids_missing = [str(id) for id in all_ids if id not in ids_found]
# pmids_missing = {}
# for id in ids_missing:
#     ref = refs[df["id"] == int(id)][0]

#     if isinstance(ref, str) and 'pubmed' in ref:
#         paper_id = get_paper_id(ref)
#         # print(id, ref, paper_id)
#         pmids_missing[paper_id] = id
# s = ",".join(list(pmids_missing.keys()))
# # !python -m pubmed2pdf pdf --pmids="{s}"

# # rename each pdf file in pubmed2pdf to its id
# pubmed_papers_dir = pathlib.Path("../pubmed2pdf")
# papers_downloaded = os.listdir(pubmed_papers_dir)
# for paper in papers_downloaded:
#     paper_id = paper.split(".")[0]
#     paper_id = pmids_missing[paper_id]
#     os.rename(
#         join(pubmed_papers_dir, paper),
#         join(pubmed_papers_dir, f"{paper_id}.pdf"),
#     )


def extract_texts_from_pdf(ids, papers_dir=papers_dir):
    for id in tqdm(ids):
        paper_file = join(papers_dir, str(id) + ".pdf")
        if pathlib.Path(paper_file).exists():
            # extract pdf with fitz
            with fitz.open(paper_file) as doc:  # open document
                text = chr(12).join([page.get_text() for page in doc])
                text = text.replace("-\n", "")
                pathlib.Path(join(papers_dir, str(id) + ".txt")).write_bytes(
                    text.encode()
                )


def download_gsheet(
    papers_dir=papers_dir,
    fill_href=True,
    run_pdf_checks=True,
    run_data_checks=True,
    run_patient_total_checks=True,
):
    def remove_html_tags(text):
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text).strip()

    df = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1x-epUl-KidVMI-AhMvT-M6nSULvE5JwNi7uF-vI1rbU/gviz/tq?tqx=out:csv&sheet=main",
        skiprows=0,
    )
    df.columns = list(map(remove_html_tags, df.columns))
    if fill_href:
        df["ref_href"] = pubmed.get_updated_refs(df)

    # check that found papers are present
    ids_with_paper = df[df["found_paper"] == 1].id.astype(int).values
    ids_found = sorted(
        [
            int(x.replace(".pdf", ""))
            for x in os.listdir(papers_dir)
            if x.endswith(".pdf")
        ]
    )

    if run_pdf_checks:
        for paper_id in ids_with_paper:
            if paper_id in ids_found:
                continue
            else:
                print("should have paper", paper_id)

        for paper_id in ids_found:
            if paper_id in ids_with_paper:
                continue
            else:
                print("ID", paper_id, "in local pdfs but not in main.csv")
                idx = df[df.id == paper_id].index[0]
                # print(df.loc[idx, "found_paper"])
                df.loc[idx, "found_paper"] = 1

    if run_patient_total_checks:
        idxs = (
            (df.num_male_corrected >= 1)
            & (df.num_female_corrected >= 1)
            & (df.num_total_corrected <= 0)
        )

        df.loc[idxs, "num_total_corrected"] = (
            df.loc[idxs, "num_male_corrected"] + df.loc[idxs, "num_female_corrected"]
        )

    # run automatic df checks
    if run_data_checks:
        test_dataframe(df)

    return df, ids_with_paper


def extract_on_subsets(
    x: str,
    llm,
    content_str: str = "## Return each table and table caption in the following text. Make sure it is a valid table and not just a description of a tables. Format the table in markdown. If None are found, return None.\n\n{input}",
    subset_len_tokens=7500,
    max_calls=3,
    frequency_penalty=0.01,
) -> str:
    """Extract on subsets of x, each of length subset_len_tokens, and return the concatenation of all tables found.
    Potential issue: if a table is split across two subsets, it will not be found.
    """
    messages = [
        {
            "role": "user",
            "content": content_str,
        }
    ]
    subset_len_chars = subset_len_tokens * 4

    tables_str = ""
    subset_num = 0

    while subset_num < max_calls:
        subset = x[subset_num * subset_len_chars : (subset_num + 1) * subset_len_chars]

        # if approx_tokens < 6000:
        messages[0]["content"] = content_str.format(input=subset)
        msg = llm(
            messages,
            temperature=0.0,
            verbose=True,
            frequency_penalty=frequency_penalty,
        )

        subset_num += 1

        tables_str += msg + "\n"

        # next segment should have atleast 0.3 * subset_len_chars_left
        if len(x) < (subset_num + 0.3) * subset_len_chars:
            break

    return tables_str


def extract_on_pages(
    paper_file: str,
    llm,
    # content_str: str="## Return each table and table caption in the following text. Make sure it is a valid table and not just a description of a tables. Format the table in markdown. If None are found, return None.\n\n{input}",
    # content_str: str="## Return the input with cleaner formatting and use markdown formatting for tables.\n\n**Input**: {input}",
    content_str="## Repeat the entire input between <INPUT> and </INPUT>, but remove unnecessary newlines and format tables in markdown.\n\n<INPUT>\n{input}\n</INPUT>",
    max_calls: int = 10,
    frequency_penalty=0,
) -> str:
    """Extract on subsets of x, each of length subset_len_tokens, and return the concatenation of all tables found.
    Potential issue: if a table is split across two subsets, it will not be found.
    """
    messages = [
        {
            "role": "user",
            "content": content_str,
        }
    ]
    # get texts for each page
    with fitz.open(paper_file) as doc:  # open document
        texts = [page.get_text() for page in doc]
        texts = [text.replace("-\n", "") for text in texts]
        for i, text in enumerate(texts):
            if "REFERENCES" in text:
                texts[i] = text[: text.index("REFERENCES")]
                texts = texts[: i + 1]

    # split each text into 2 equal pieces
    # texts_new = []
    # for text in texts:
    #     if len(text) > 400:
    #         texts_new.append(text[:len(text)//2])
    #         texts_new.append(text[len(text)//2:])
    #     else:
    #         texts_new.append(text)
    # texts = texts_new

    for text in texts:
        print(len(text))

    tables_str = ""
    subset_num = 0

    for i in tqdm(range(min(len(texts), max_calls))):
        messages[0]["content"] = content_str.format(input=texts[i])
        msg = llm(
            messages,
            temperature=0.0,
            verbose=True,
            frequency_penalty=frequency_penalty,
        )

        tables_str += msg + "\n"

    return tables_str


def test_dataframe(df):
    """Checks that the dataframe has the correct format and values"""
    assert len(df) == 690, "df should have 690 rows"

    # check if string is an integer
    def _is_int(x):
        try:
            int(x)
            return True
        except:
            return False

    # check that values are integers or Unk
    for col in df.columns:
        # if col.startswith("num_") and col.endswith("_corrected"):
        if col in ["num_male_corrected", "num_female_corrected", "num_total_corrected"]:
            vals = df[col][df[col].notna()].values
            for val in vals:
                assert val in {"Unk", "-"} or _is_int(
                    val
                ), f"{col} has {val} which is not an int or Unk"

    # check some individual rows
    row = df[df.id == 10470].iloc[0]
    assert row.full_title_en == "VIRSTA Score"
    assert row.short_description_en == "IE risk."
    assert row.ref_href == "https://pubmed.ncbi.nlm.nih.gov/26916042/"
    assert row.num_male_corrected == 1295
    assert row.num_female_corrected == 713
    assert row.num_total_corrected == 2008

    row = df[df.id == 10210].iloc[0]
    assert row.full_title_en == "PREVAIL Model for Prostate Cancer Survival"
    assert row.short_description_en == "Overall survival in metastatic prostate cancer."
    assert row.ref_href == "https://www.ncbi.nlm.nih.gov/pubmed/30202945"
    assert row.num_male_corrected == -1
