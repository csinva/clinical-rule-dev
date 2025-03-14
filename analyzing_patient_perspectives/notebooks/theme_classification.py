import pandas as pd
import os
import os.path
from os.path import join
import numpy as np
import imodelsx
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import data
import imodelsx.viz
import joblib
import dvu


def numbered_list(classes):
    # return '\n'.join([f'Class {i+1}. {c}' for i, c in enumerate(classes)])
    return '\n'.join([f'{i+1}. {c}' for i, c in enumerate(classes)])


classification_prompt = '''### You are given a question, response, and a numbered list of themes below.

Question: {question}

Response: {response}

Themes:
{classes_as_numbered_list}

### Which of the themes does the response above belong to? Return the theme number.

Answer:'''


def get_valid_question_nums(qs, responses_df, themes_df):
    valid_question_nums = []
    for question_num in tqdm(range(len(qs)), position=0):

        question, responses, theme_dict = data.get_data_for_question_single_site(
            question_num=question_num, qs=qs, responses_df=responses_df, themes_df=themes_df)

        # remove weird hanging error
        child_neurologist_in_response = False
        for i in range(len(responses)):
            if isinstance(responses.values[i], str) and 'Child neurologist' in responses.values[i]:
                child_neurologist_in_response = True
        if child_neurologist_in_response:
            continue

        # print(theme_dict)
        # valid only if there are multiple themes
        if len(theme_dict) > 1:

            # valid only if the number of theme values == number of non-null responses
            if sum(theme_dict.values()) == sum(responses.notna()):
                valid_question_nums.append(question_num)

            # assert sum(theme_dict.values()) == sum(responses.notna(
            # )), f'{sum(theme_dict.values())} != {sum(responses.notna())}, theme_dict: {theme_dict}, responses: {responses}\nquestion {question_num}: {question}'
    print('num valid qs', len(valid_question_nums))
    return valid_question_nums


def get_classifications(llm, valid_question_nums, qs, responses_df, themes_df):
    classifications = defaultdict(list)
    for question_num in tqdm(valid_question_nums, position=0):

        question, responses, theme_dict = data.get_data_for_question_single_site(
            question_num=question_num, qs=qs, responses_df=responses_df, themes_df=themes_df)

        for response_num in tqdm(range(len(responses)), position=1):
            response = responses.values[response_num]

            if not pd.isna(response):
                prompt = classification_prompt.format(
                    question=question,
                    classes_as_numbered_list=numbered_list(theme_dict.keys()),
                    response=response,
                )
                ans = llm(prompt, verbose=False)
                classifications[question_num].append(ans)
    return classifications


def compute_and_save_eval_mat(classifications, valid_question_nums, qs, responses_df, themes_df, site, checkpoint):
    def _process_classifications(x):
        # note, only works with <10 classes
        for i in range(1, 10):
            if x.startswith(str(i)):
                return i
        else:
            return 1

    diffs = []
    class_count_pred = []
    class_count_gt = []
    for question_num in valid_question_nums:
        question, responses, theme_dict = data.get_data_for_question_single_site(
            question_num=question_num, qs=qs, responses_df=responses_df, themes_df=themes_df)

        answer_counts = theme_dict.values()
        class_counts = pd.Series(classifications[question_num]).apply(
            _process_classifications).value_counts()

        # convert value_counts to dict
        answer_counts_dict = {i + 1: round(v)
                              for i, v in enumerate(theme_dict.values())}
        class_counts_dict = {k: round(v) for k, v in class_counts.items()}
        for i in answer_counts_dict.keys():
            if i not in class_counts_dict:
                class_counts_dict[i] = 0

        # print(answer_counts_dict)
        # print(class_counts_dict)
        # print()

        class_count_pred += list(class_counts_dict.values())
        class_count_gt += list(answer_counts_dict.values())

        diffs.append(np.sum(np.abs(np.array(list(answer_counts_dict.values())) -
                                   np.array(list(class_counts_dict.values())))))

    n = max(class_count_gt + class_count_pred)
    mat = np.zeros((n, n))
    for i, j in zip(class_count_pred, class_count_gt):
        mat[i - 1, j - 1] += 1
    joblib.dump(mat, join(data.PROCESSED_DIR,
                f'classification_confusion_matrix_{site}_{checkpoint.split("/")[-1]}.pkl'))
