from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'label_name': ["categorization___chief_complaint",
                   "categorization___specialty",
                   "categorization___purpose",
                   "categorization___system",
                   "categorization___disease"],
    # 'input_text': ['raw_text', 'description'],
    # 'model_name': ['random_forest', 'logistic'],
    'input_text': ['description'],
    # 'model_name': ['bert-base-uncased'],  # , 'aug-linear'],
    'model_name': ['zero-shot'],

    'save_dir': [join(repo_dir, 'results', 'classification')],
    # pass binary values with 0/1 instead of the ambiguous strings True/False
    'use_cache': [1],
}

params_coupled_dict = {
}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'notebooks_llm', '02_classification.py'),
    actually_run=True,
    n_cpus=1
)
