import os
from os.path import join as oj
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import data

NUM_PATIENTS = 12044
DATA_DIR = 'data_pecarn/Datasets'


def get_data(use_processed=False, frac_missing_allowed=0.05, processed_file='processed/df_pecarn.pkl', dummy=False, impute_feats=True):
    '''Run all the preprocessing
    
    Params
    ------
    use_processed: bool, optional
        determines whether to load df from cached pkl
    save_processed: bool, optional
        if not using processed, determines whether to save the df
    '''
    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        print('computing pecarn preprocessing...')
        df_features = get_features()  # read all features into df
        df_outcomes = get_outcomes()  # 2 outcomes: iai, and iai_intervention
        df = pd.merge(df_features, df_outcomes, on='id', how='left')
        df = rename_values(df)  # rename the features by their meaning
        
        # drop cols with vals missing this percent of the time
        df = df.dropna(axis=1, thresh=(1 - frac_missing_allowed) * NUM_PATIENTS)
        
        # delete repeat columns
        '''
        keys = list(df.keys())
        keys_to_remove = [k for k in keys if 'Repeat_instance' in k]
        df = df.drop(labels=keys_to_remove, axis=1)
        '''
        if impute_feats:
            df = impute(df)  # impute and fill
        df = data.add_cv_split(df, dset='pecarn')
        if dummy:
            df = data.to_dummies(df)
        df['dset'] = 'pecarn'
        
        # save
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        df.to_pickle(processed_file)
        return df


def get_features(processed_file='processed/df_pecarn_features.pkl'):
    '''Read all features into df
    
    Returns
    -------
    features: pd.DataFrame
    '''
    
    if os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    
    fnames = sorted([fname for fname in os.listdir(DATA_DIR)
                     if 'csv' in fname
                     and not 'formats' in fname
                     and not 'form6' in fname])  # remove outcome
    # feature_names = [fname[:-4].replace('form', '').replace('-', '_') for fname in fnames]
    # demographics = pd.read_csv('iaip_data/Datasets/demographics.csv')
    # print(fnames)
    r = {}
    for fname in tqdm(fnames):
        df = pd.read_csv(oj(DATA_DIR, fname), engine='python')
        df.rename(columns={'SubjectID': 'id'}, inplace=True)
        df.rename(columns={'subjectid': 'id'}, inplace=True)
        assert ('id' in df.keys())
        r[fname] = df

    df = r[fnames[0]]
    fnames_small = [fname for fname in fnames
                    if 'form1' in fname
                    or 'form2' in fname
                    or 'form4' in fname
                    or 'form5' in fname
                    or 'form7' in fname
                    ]
    for i, fname in tqdm(enumerate(fnames_small)):
        df2 = r[fname].copy()
        df2 = df2.drop_duplicates(subset=['id'], keep='last')  # if subj has multiple entries, only keep first
        rename_dict = {
            key: key  # + '_' + fname[:-4].replace('form', '')
            for key in df2.keys()
            if not key == 'id'
        }
        df2.rename(columns=rename_dict, inplace=True)
        df = df.set_index('id').combine_first(df2.set_index('id')).reset_index()  # don't save duplicate columns

    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    df.to_pickle(processed_file)
    # df.to_pickle(oj(pdir, 'features.pkl'))
    return df


def get_outcomes():
    """Read in the outcomes

    Returns
    -------
    outcomes: pd.DataFrame
        iai (has 761 positives)
        iai_intervention (has 203 positives)
    """
    form4abdangio = pd.read_csv(oj(DATA_DIR, 'form4bother_abdangio.csv')).rename(columns={'subjectid': 'id'})
    # form6a = pd.read_csv(oj(DATA_DIR, 'form6a.csv')).rename(columns={'subjectid': 'id'})
    form6b = pd.read_csv(oj(DATA_DIR, 'form6b.csv')).rename(columns={'SubjectID': 'id'})
    form6c = pd.read_csv(oj(DATA_DIR, 'form6c.csv')).rename(columns={'subjectid': 'id'})

    # (6b) Intra-abdominal injury diagnosed in the ED/during hospitalization by any diagnostic method
    # 1 is yes, 761 have intra-abdominal injury
    # 2 is no -> remap to 0, 841 without intra-abdominal injury

    def get_ids(form, keys):
        '''Returns ids for which any of the keys is 1
        '''
        ids_all = set()
        for key in keys:
            ids = form.id.values[form[key] == 1]
            for i in ids:
                ids_all.add(i)
        return ids_all

    ids_iai = get_ids(form6b, ['IAIinED1']) # form6b.id[form6b['IAIinED1'] == 1]

    # print(form4abdangio.keys())
    ids_allangio = get_ids(form4abdangio, ['AbdAngioVessel'])
    # print('num in 4angio', len(ids_allangio))
    # print(form6a.keys())
    # ids_alla = get_ids(form6a, ['DeathCause'])
    # print('num in a', len(ids_alla))
    # print(form6b.keys())
    ids_allb = get_ids(form6b, ['IVFluids', 'BldTransfusion'])
    # print('num in b', len(ids_allb))
    # print(form6c.keys())
    ids_allc = get_ids(form6c, ['IntervenDurLap'])
    # print('num in c', len(ids_allc))
    ids = ids_allb.union(ids_allangio).union(ids_allc)

    ids_iai_np = np.array(list(ids_iai)) - 1
    ids_np = np.array(list(ids)) - 1

    iai = np.zeros(NUM_PATIENTS).astype(np.int)
    iai[ids_iai_np] = 1
    iai_intervention = np.zeros(NUM_PATIENTS).astype(np.int)
    iai_intervention[ids_np] = 1

    df_iai = pd.DataFrame.from_dict({
        'id': np.arange(1, NUM_PATIENTS + 1),
        'iai': iai,
        'iai_intervention': iai_intervention
    })
    return df_iai


def rename_values(df):
    '''Map values to meanings
    Rename some features
    Compute a couple new features
    set types of 
    '''

    race = {
        1: 'American Indian or Alaska Native',
        2: 'Asian',
        3: 'Black or African American',
        4: 'Native Hawaiian or other Pacific Islander',
        5: 'White',
        6: 'unknown', # stated as unknown
        7: 'unknown' # other
    }
    moi = {
        1: 'Motor vehicle collision',
        2: 'Fall from an elevation',
        3: 'Fall down stairs',
        4: 'Pedestrian/bicyclist struck by moving vehicle',
        5: 'Bike collision/fall',
        6: 'Motorcycle/ATV/Scooter collision',
        7: 'Object struck abdomen',
        8: 'unknown', # unknown mechanism,
        9: 'unknown', # other mechanism
        10: 'unknown' # physician did not answer
    }
    abdTenderDegree = {
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'unknown'
    }
    df.RACE = df.RACE.map(race)
    df['MOI'] = df.RecodedMOI.map(moi)
    df = df.drop(columns=['RecodedMOI'])
    
    df['AbdTenderDegree'] = df.AbdTenderDegree.fillna(1).map(abdTenderDegree)
    df = df.rename(columns={'RACE': 'Race_orig', 
                            'SEX': 'Sex', 
                            'HISPANIC_ETHNICITY': 'Hispanic',
                            'ageinyrs': 'Age'
                           })

    # set types of these variables to categorical
    ks_categorical = ['Sex', 'Race_orig', 'Hispanic',
                      'VomitWretch', 'MOI', 'ThoracicTender', 'ThoracicTrauma',
                      'DecrBreathSound', 'AbdDistention', 'AbdTenderDegree',
                      'AbdTrauma', 'SeatBeltSign', 'DistractingPain',
                      'AbdomenPain']
    for k in ks_categorical:
        df[k] = df[k].astype(str)    
    binary = {
        0: 'no',
        1: 'yes',
        False: 'no',
        True: 'yes',
        'unknown': 'unknown'
    }
    
    df['AbdomenPain'] = df['AbdomenPain'].replace('3.0', 'other')
    df["Hispanic"]= (df['Hispanic'] == '-1').map(binary) # note: -1 is Hispanice (0 is not, 1 is unknown)
    
    
    # rename vars to values
    ks_remap = ['VomitWretch', 'MOI', 
                'ThoracicTender', 'ThoracicTrauma', 
                'DecrBreathSound', 'AbdDistention', 'AbdTenderDegree',
                'AbdTrauma', 'SeatBeltSign', 
                'DistractingPain', 'AbdomenPain']
    for k in ks_remap:
        vals = df[k].values
        is_na = df[k].isna()
        uniques = np.unique(vals).astype(np.str)
        contains_nan = np.sum(is_na) > 0
        if contains_nan and uniques.size in [4, 5] or ~contains_nan and uniques.size in [3, 4]:
            if '1.0' in uniques and '2.0' in uniques and ('3.0' in uniques or 'other' in uniques):
                df[k] = df[k].map({
                    '1.0': 'yes',
                    '2.0': 'no',
                    '3.0': 'unknown',
                    '4.0': 'unknown',
                    'other': 'other',
                    np.nan: 'unknown',
                })
    df = data.derived_feats(df)
    return df


def impute(df: pd.DataFrame):
    """Returns df with imputed features
    """

    # fill in values for some vars from NaN -> Unknown
    df['AbdTrauma'] = df['AbdTrauma'].fillna('unknown')


    # pandas impute missing values with median
    df = df.fillna(df.median())
    df.GCSScore = df.GCSScore.fillna(df.GCSScore.median())
    return df