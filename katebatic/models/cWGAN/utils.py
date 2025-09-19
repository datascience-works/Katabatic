## Dataloader.py

import pandas as pd
import numpy as np

import logging

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, average_precision_score, \
    mean_squared_error, r2_score, brier_score_loss
from sklearn.utils import resample
from scipy.stats import pearsonr

import random
import string
import datetime

from typing import Tuple




def get_datasets(names_only: bool = False):
    DATASET_DICT = {
        'statlog': load_statlog,
        'bank': load_bank,
        'thomas': load_thomas,
        'pakdd': load_pakdd,
        'taiwan': load_taiwan,
        'homeeq': load_homeeq,
        'lendingcluba': load_lendingcluba,
        'lendingclubb': load_lendingclubb,
        'gmc': load_gmc,
        'dmc_05': load_dmc05,
        'dmc_10': load_dmc10,
        'coil2k': load_coil2k,
        'adult': load_adult,
        'statlog_australian': load_statlog_australian
    }

    if names_only:
        return list(DATASET_DICT.keys())
    else:
        return DATASET_DICT


def get_dataset_setting(dataset: str) -> str:
    settings = {'statlog': 'Credit scoring',
                'bank': 'Marketing',
                'homeeq': 'Credit scoring',
                'dmc_05': 'Profitability scoring',
                'dmc_10': 'Response modeling',
                'coil2k': 'Response modeling',
                'adult': 'Income prediction'}
    setting = settings[dataset]
    return setting


def get_dataset_source(dataset: str) -> str:
    sources = {'statlog': 'UCI MLR',
               'bank': 'UCI MLR',
               'homeeq': 'Baesens et al.',
               'dmc_05': 'DMC 2005',
               'dmc_10': 'DMC 2010',
               'coil2k': 'UCI MLR',
               'adult': 'UCI MLR'}
    source = sources[dataset]
    return source


def load_data(dataset: str):
    logging.debug(f'Dataloader: Loading {dataset}')

    dataset_dict = get_datasets()

    logging.debug(f'Dataloader: Loaded available datasets.')

    if dataset in dataset_dict.keys():
        func = dataset_dict[dataset]
        df, cat_cols, num_cols, target_col = func()
    else:
        logging.error(f'Dataloader: Dataset {dataset} not found.')
        raise ValueError(f'Dataloader: Dataset "{dataset}" not found.')

    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes.astype('category'))

    logging.info(f'Dataloader: Loaded dataset: {dataset}. Returning data.')

    return df, cat_cols, num_cols, target_col


# ## Preprocessing

# #### statlog
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

def load_statlog():
    path = 'Datasets/Raw/UCI_statlog_german_credit_data_data_set/german.data'

    col_names = ['Status_checking_account', 'Duration_months', 'Credit_history', 'Purpose',
                 'Credit_amount', 'Savings_account_bonds', 'Present_employment_since',
                 'Instalment_rate_percent_of_income', 'Personal_status_sex', 'Other_debtors_guarantors',
                 'Present_residence_since', 'Property', 'Age_years', 'Other_instalment_plans',
                 'Housing', 'Number_of_existing_credits', 'Job', 'Dependants', 'Telephone',
                 'Foreign_worker', 'Status_loan']

    cat_cols = ['Status_checking_account', 'Credit_history', 'Purpose',
                'Savings_account_bonds', 'Present_employment_since', 'Personal_status_sex',
                'Other_debtors_guarantors', 'Property', 'Other_instalment_plans', 'Housing',
                'Job', 'Telephone', 'Foreign_worker']
    num_cols = ['Duration_months', 'Credit_amount', 'Instalment_rate_percent_of_income',
                'Present_residence_since', 'Age_years', 'Number_of_existing_credits', 'Dependants']

    target_col = 'Status_loan'

    df = pd.read_csv(path, sep=' ', header=None, index_col=False,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    df[target_col] = df[target_col] - 1

    return df, cat_cols, num_cols, target_col


# #### statlog australian
# http://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29

def load_statlog_australian():
    path = 'Datasets/Raw/UCI_statlog_australian_credit_data_data_set/australian.dat'

    col_names = [f'A{i}' for i in range(1, 16)]

    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    num_cols = [c for c in col_names if c not in cat_cols and c != 'A15']

    target_col = 'A15'

    df = pd.read_csv(path, sep=' ', header=None, index_col=False,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    return df, cat_cols, num_cols, target_col


# #### Thomas2002
# L.C. Thomas, D.B. Edelman, J.N. Crook, Credit Scoring and its Applications, SIAM, Philadelphia, 2002.
# https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/5.%20thomas/Loan%20Data.csv

def load_thomas():
    path = 'Datasets/Raw/Thomas_et_al_data_set/Loan Data.csv'

    cat_cols = ['PHON', 'AES', 'RES']

    target_col = 'BAD'

    df = pd.read_csv(path, sep=';', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    return df, cat_cols, num_cols, target_col


# ####  PAKDD2010
# https://github.com/JLZml/Credit-Scoring-Data-Sets/blob/master/2.%20PAKDD%202009%20Data%20Mining%20Competition/PAKDD%202010.zip
# http://sede.neurotech.com.br:443/PAKDD2009/

def load_pakdd():
    path = 'Datasets/Raw/PAKDD2010_data_set/PAKDD2010_Modeling_Data.txt'

    columns = ["ID_CLIENT", "CLERK_TYPE", "PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "QUANT_ADDITIONAL_CARDS",
               "POSTAL_ADDRESS_TYPE", "SEX", "MARITAL_STATUS", "QUANT_DEPENDANTS", "EDUCATION_LEVEL", "STATE_OF_BIRTH",
               "CITY_OF_BIRTH", "NACIONALITY", "RESIDENCIAL_STATE", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH",
               "FLAG_RESIDENCIAL_PHONE", "RESIDENCIAL_PHONE_AREA_CODE", "RESIDENCE_TYPE", "MONTHS_IN_RESIDENCE",
               "FLAG_MOBILE_PHONE", "FLAG_EMAIL", "PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES", "FLAG_VISA",
               "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS", "QUANT_BANKING_ACCOUNTS",
               "QUANT_SPECIAL_BANKING_ACCOUNTS", "PERSONAL_ASSETS_VALUE", "QUANT_CARS", "COMPANY", "PROFESSIONAL_STATE",
               "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "FLAG_PROFESSIONAL_PHONE", "PROFESSIONAL_PHONE_AREA_CODE",
               "MONTHS_IN_THE_JOB", "PROFESSION_CODE", "OCCUPATION_TYPE", "MATE_PROFESSION_CODE",
               "MATE_EDUCATION_LEVEL", "FLAG_HOME_ADDRESS_DOCUMENT", "FLAG_RG", "FLAG_CPF", "FLAG_INCOME_PROOF",
               "PRODUCT", "FLAG_ACSP_RECORD", "AGE", "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3", "TARGET_BAD"]

    cat_cols = ['PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS',
                'STATE_OF_BIRTH', 'NACIONALITY', 'RESIDENCIAL_STATE', 'FLAG_RESIDENCIAL_PHONE',
                'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE', 'FLAG_EMAIL', 'FLAG_VISA', 'FLAG_MASTERCARD',
                'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS',
                'QUANT_SPECIAL_BANKING_ACCOUNTS', 'COMPANY', 'PROFESSIONAL_STATE', 'FLAG_PROFESSIONAL_PHONE',
                'PROFESSIONAL_PHONE_AREA_CODE', 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE',
                'MATE_EDUCATION_LEVEL', 'PRODUCT']

    num_cols = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'PERSONAL_ASSETS_VALUE', 'AGE', 'MONTHS_IN_RESIDENCE',
                'QUANT_DEPENDANTS', 'QUANT_CARS', 'MONTHS_IN_THE_JOB']

    target_col = 'TARGET_BAD'

    drop_cols = ['CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'PROFESSIONAL_CITY',
                 'PROFESSIONAL_BOROUGH', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3', 'FLAG_HOME_ADDRESS_DOCUMENT',
                 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'FLAG_ACSP_RECORD', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS',
                 'EDUCATION_LEVEL', 'FLAG_MOBILE_PHONE']

    df = pd.read_csv(path, sep='\t',
                     index_col='ID_CLIENT', encoding='unicode_escape',
                     header=None, names=columns,
                     dtype={col: 'category' for col in cat_cols}).drop(drop_cols, axis=1)

    return df, cat_cols, num_cols, target_col


# #### Taiwan
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

def load_taiwan():
    path = 'Datasets/Raw/UCI_taiwan_default_of_credit_card_clients_data_set/default of credit card clients.csv'

    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    target_col = 'default payment next month'

    df = pd.read_csv(path, index_col=0,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    return df, cat_cols, num_cols, target_col


# #### bank
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

def load_bank():
    path = 'Datasets/Raw/UCI_bank_marketing_data_set/bank-additional-full.csv'

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'poutcome']
    num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

    target_col = 'y'

    df = pd.read_csv(path, sep=';', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    df['y'] = np.where(df['y'] == 'yes', 1, 0)

    return df, cat_cols, num_cols, target_col


# #### homeeq
# http://www.creditriskanalytics.net/datasets-private2.html

def load_homeeq():
    path = 'Datasets/Raw/CREDITRISKANALYTICS_home_equity_data_set/hmeq.csv'

    cat_cols = ['REASON', 'JOB']
    num_cols = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG',
                'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

    target_col = 'BAD'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols})

    return df, cat_cols, num_cols, target_col


# #### lending club 3a
# https://www.lendingclub.com/info/download-data.action

def load_lendingcluba():
    path = 'Datasets/Raw/Lending_Club_data_sets/LoanStats3a.csv'

    cat_cols = ['debt_settlement_flag', 'term', 'pub_rec_bankruptcies', 'verification_status', 'loan_status',
                'home_ownership', 'pub_rec', 'grade', 'emp_length', 'purpose', 'sub_grade', 'title']
    num_cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'annual_inc', 'revol_util', 'dti',
                'installment', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc', 'total_pymnt', 'total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'last_pymnt_amnt']

    # many NANs and only one unique value present, or very few
    drop_cols = ['mths_since_last_delinq', 'mths_since_last_record', 'next_pymnt_d', 'settlement_term',
                 'settlement_amount', 'settlement_date', 'settlement_status', 'debt_settlement_flag_date',
                 'settlement_percentage', 'id', 'tot_coll_amt', 'sec_app_num_rev_accts', 'sec_app_open_act_il',
                 'sec_app_revol_util', 'sec_app_open_acc', 'sec_app_mort_acc', 'sec_app_inq_last_6mths',
                 'sec_app_earliest_cr_line', 'sec_app_chargeoff_within_12_mths', 'revol_bal_joint', 'total_bc_limit',
                 'total_bal_ex_mort', 'tot_hi_cred_lim', 'percent_bc_gt_75', 'pct_tl_nvr_dlq', 'num_tl_op_past_12m',
                 'num_tl_90g_dpd_24m', 'total_il_high_credit_limit', 'sec_app_collections_12_mths_ex_med',
                 'sec_app_mths_since_last_major_derog', 'hardship_type', 'member_id', 'url',
                 'mths_since_last_major_derog', 'annual_inc_joint', 'hardship_last_payment_amount',
                 'hardship_payoff_balance_amount', 'orig_projected_additional_accrued_interest', 'hardship_loan_status',
                 'hardship_dpd', 'hardship_length', 'payment_plan_start_date', 'hardship_end_date',
                 'hardship_start_date', 'hardship_amount', 'deferral_term', 'hardship_status', 'hardship_reason',
                 'num_tl_30dpd', 'verification_status_joint', 'num_tl_120dpd_2m', 'num_rev_tl_bal_gt_0', 'inq_last_12m',
                 'total_cu_tl', 'inq_fi', 'total_rev_hi_lim', 'all_util', 'max_bal_bc', 'open_rv_24m',
                 'acc_open_past_24mths', 'open_rv_12m', 'total_bal_il', 'mths_since_rcnt_il', 'open_il_24m',
                 'open_il_12m', 'open_act_il', 'open_acc_6m', 'tot_cur_bal', 'il_util', 'avg_cur_bal', 'bc_open_to_buy',
                 'bc_util', 'num_rev_accts', 'num_op_rev_tl', 'dti_joint', 'num_bc_tl', 'num_bc_sats',
                 'num_actv_rev_tl', 'num_actv_bc_tl', 'num_accts_ever_120_pd', 'mths_since_recent_revol_delinq',
                 'mths_since_recent_inq', 'mths_since_recent_bc_dlq', 'mths_since_recent_bc', 'mort_acc',
                 'mo_sin_rcnt_tl', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_old_rev_tl_op', 'mo_sin_old_il_acct', 'num_sats',
                 'num_il_tl',
                 'policy_code', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med',
                 'initial_list_status', 'application_type', 'hardship_flag', 'chargeoff_within_12_mths',
                 'acc_now_delinq', 'tax_liens', 'delinq_amnt',
                 'emp_title', 'zip_code', 'addr_state', 'last_pymnt_d', 'last_credit_pull_d', 'desc',
                 'earliest_cr_line', 'issue_d']

    target_col = 'delinq_2yrs'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols}
                     ).drop(drop_cols, axis=1)

    df['pub_rec_bankruptcies'] = (df['pub_rec_bankruptcies'].astype(float) > 0).astype('category')
    df['pub_rec'] = (df['pub_rec'].astype(float) > 0).astype('category')

    df['title'] = np.where(df['title'].isin(df.title.value_counts()[df.title.value_counts() > 100].index.values),
                           df['title'], 'OTHER')
    df['title'] = df['title'].astype('category')

    df['int_rate'] = df['int_rate'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)
    df['revol_util'] = df['revol_util'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)

    df = df.loc[~df[target_col].isna()]
    df[target_col] = (df[target_col] > 0).astype(int)

    return df, cat_cols, num_cols, target_col


# #### lending club 3b
# https://www.lendingclub.com/info/download-data.action

def load_lendingclubb():
    path = 'Datasets/Raw/Lending_Club_data_sets/LoanStats3b.csv'

    cat_cols = ['debt_settlement_flag', 'term', 'pub_rec_bankruptcies', 'verification_status', 'loan_status',
                'home_ownership', 'pub_rec', 'grade', 'emp_length', 'purpose', 'sub_grade', 'title']
    num_cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'annual_inc', 'revol_util', 'dti',
                'installment', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc', 'total_pymnt', 'total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'last_pymnt_amnt']

    # many NANs and only one unique value present, or very few
    drop_cols = ['mths_since_last_delinq', 'mths_since_last_record', 'next_pymnt_d', 'settlement_term',
                 'settlement_amount', 'settlement_date', 'settlement_status', 'debt_settlement_flag_date',
                 'settlement_percentage', 'id', 'tot_coll_amt', 'sec_app_num_rev_accts', 'sec_app_open_act_il',
                 'sec_app_revol_util', 'sec_app_open_acc', 'sec_app_mort_acc', 'sec_app_inq_last_6mths',
                 'sec_app_earliest_cr_line', 'sec_app_chargeoff_within_12_mths', 'revol_bal_joint', 'total_bc_limit',
                 'total_bal_ex_mort', 'tot_hi_cred_lim', 'percent_bc_gt_75', 'pct_tl_nvr_dlq', 'num_tl_op_past_12m',
                 'num_tl_90g_dpd_24m', 'total_il_high_credit_limit', 'sec_app_collections_12_mths_ex_med',
                 'sec_app_mths_since_last_major_derog', 'hardship_type', 'member_id', 'url',
                 'mths_since_last_major_derog', 'annual_inc_joint', 'hardship_last_payment_amount',
                 'hardship_payoff_balance_amount', 'orig_projected_additional_accrued_interest', 'hardship_loan_status',
                 'hardship_dpd', 'hardship_length', 'payment_plan_start_date', 'hardship_end_date',
                 'hardship_start_date', 'hardship_amount', 'deferral_term', 'hardship_status', 'hardship_reason',
                 'num_tl_30dpd', 'verification_status_joint', 'num_tl_120dpd_2m', 'num_rev_tl_bal_gt_0', 'inq_last_12m',
                 'total_cu_tl', 'inq_fi', 'total_rev_hi_lim', 'all_util', 'max_bal_bc', 'open_rv_24m',
                 'acc_open_past_24mths', 'open_rv_12m', 'total_bal_il', 'mths_since_rcnt_il', 'open_il_24m',
                 'open_il_12m', 'open_act_il', 'open_acc_6m', 'tot_cur_bal', 'il_util', 'avg_cur_bal', 'bc_open_to_buy',
                 'bc_util', 'num_rev_accts', 'num_op_rev_tl', 'dti_joint', 'num_bc_tl', 'num_bc_sats',
                 'num_actv_rev_tl', 'num_actv_bc_tl', 'num_accts_ever_120_pd', 'mths_since_recent_revol_delinq',
                 'mths_since_recent_inq', 'mths_since_recent_bc_dlq', 'mths_since_recent_bc', 'mort_acc',
                 'mo_sin_rcnt_tl', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_old_rev_tl_op', 'mo_sin_old_il_acct', 'num_sats',
                 'num_il_tl',
                 'policy_code', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med',
                 'initial_list_status', 'application_type', 'hardship_flag', 'chargeoff_within_12_mths',
                 'acc_now_delinq', 'tax_liens', 'delinq_amnt',
                 'emp_title', 'zip_code', 'addr_state', 'last_pymnt_d', 'last_credit_pull_d', 'desc',
                 'earliest_cr_line', 'issue_d']

    target_col = 'delinq_2yrs'

    df = pd.read_csv(path, sep=',', index_col=False,
                     dtype={col: 'category' for col in cat_cols}
                     ).drop(drop_cols, axis=1)

    df['pub_rec_bankruptcies'] = (df['pub_rec_bankruptcies'].astype(float) > 0).astype('category')
    df['pub_rec'] = (df['pub_rec'].astype(float) > 0).astype('category')

    df['title'] = np.where(df['title'].isin(df.title.value_counts()[df.title.value_counts() > 100].index.values),
                           df['title'], 'OTHER')
    df['title'] = df['title'].astype('category')

    df['int_rate'] = df['int_rate'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)
    df['revol_util'] = df['revol_util'].apply(lambda x: float(x[:-1]) if not isinstance(x, np.float) else x)

    df = df.loc[~df[target_col].isna()]
    df[target_col] = (df[target_col] > 0).astype(int)

    return df, cat_cols, num_cols, target_col


# #### Give me credit
# https://www.kaggle.com/c/GiveMeSomeCredit/data?select=cs-training.csv

def load_gmc():
    path = 'Datasets/Raw/Kaggle_give_me_credit_data_set/cs-training.csv'

    cat_cols = []

    target_col = 'SeriousDlqin2yrs'

    df = pd.read_csv(path, sep=',', index_col=0,
                     dtype={col: 'category' for col in cat_cols})

    num_cols = [c for c in df.columns if c != target_col]

    return df, cat_cols, num_cols, target_col


# #### dmc05
# https://www.data-mining-cup.com/reviews/dmc-2005/


def load_dmc05():
    path = 'Datasets/Raw/DMC05_ecommerce_fraud_data_set/dmc2005_train.txt'

    cat_cols = ['B_EMAIL', 'B_TELEFON', 'B_GEBDATUM', 'FLAG_LRIDENTISCH', 'FLAG_NEWSLETTER',
                'Z_METHODE', 'Z_CARD_ART', 'Z_LAST_NAME', 'TAG_BEST', 'TIME_BEST', 'CHK_LADR',
                'CHK_RADR', 'CHK_KTO', 'CHK_CARD', 'CHK_COOKIE', 'CHK_IP', 'FAIL_LPLZ',
                'FAIL_LORT', 'FAIL_LPLZORTMATCH', 'FAIL_RPLZ', 'FAIL_RORT',
                'FAIL_RPLZORTMATCH', 'NEUKUNDE', 'DATUM_LBEST']

    num_cols = ['Z_CARD_VALID', 'WERT_BEST', 'ANZ_BEST', 'ANUMMER_01', 'ANUMMER_02',
                'ANUMMER_03', 'ANUMMER_04', 'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07',
                'ANUMMER_08', 'ANUMMER_09', 'ANUMMER_10', 'SESSION_TIME', 'ANZ_BEST_GES',
                'WERT_BEST_GES', 'MAHN_AKT', 'MAHN_HOECHST']

    target_col = 'TARGET_BETRUG'

    df = pd.read_csv(path, sep='\t', index_col='BESTELLIDENT',
                     dtype={col: 'category' for col in cat_cols})

    df['TARGET_BETRUG'] = np.where(df['TARGET_BETRUG'] == 'ja', 1, 0)

    return df, cat_cols, num_cols, target_col


# #### dmc10
# https://www.data-mining-cup.com/reviews/dmc-2010/


def load_dmc10():
    path = 'Datasets/Raw/DMC10_ecommerce_voucher_data_set/dmc2010_train.txt'

    cat_cols = ['delivpostcode', 'advertisingdatacode', 'salutation', 'title',
                'domain', 'newsletter', 'model', 'paymenttype', 'deliverytype',
                'invoicepostcode', 'voucher', 'case', 'gift', 'entry', 'points',
                'shippingcosts']
    num_cols = ['numberitems', 'weight', 'remi', 'cancel', 'used', 'w0', 'w1',
                'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'account_age']
    drop_cols = ['date', 'datecreated', 'deliverydatepromised', 'deliverydatereal']

    target_col = 'target90'

    df = pd.read_csv(path, sep=';', index_col='customernumber', parse_dates=drop_cols,
                     dtype={col: 'category' for col in cat_cols})
    df['account_age'] = (df['date'] - df['datecreated']).dt.days

    df.drop(drop_cols, axis=1, inplace=True)

    return df, cat_cols, num_cols, target_col


# #### coil2k
# http://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)


def load_coil2k():
    path = 'Datasets/Raw/UCI_coil2k_insurance_data_set/ticdata2000.txt'

    col_names = ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK',
                 'MGODPR', 'MGODOV', 'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN',
                 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG', 'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG',
                 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG', 'MBERARBO', 'MSKA',
                 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1', 'MAUT2',
                 'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575',
                 'MINK7512', 'MINK123M', 'MINKGEM', 'MKOOPKLA', 'PWAPART', 'PWABEDR',
                 'PWALAND', 'PPERSAUT', 'PBESAUT', 'PMOTSCO', 'PVRAAUT', 'PAANHANG',
                 'PTRACTOR', 'PWERKT', 'PBROM', 'PLEVEN', 'PPERSONG', 'PGEZONG', 'PWAOREG',
                 'PBRAND', 'PZEILPL', 'PPLEZIER', 'PFIETS', 'PINBOED', 'PBYSTAND',
                 'AWAPART', 'AWABEDR', 'AWALAND', 'APERSAUT', 'ABESAUT', 'AMOTSCO',
                 'AVRAAUT', 'AAANHANG', 'ATRACTOR', 'AWERKT', 'ABROM', 'ALEVEN', 'APERSONG',
                 'AGEZONG', 'AWAOREG', 'ABRAND', 'AZEILPL', 'APLEZIER', 'AFIETS',
                 'AINBOED', 'ABYSTAND', 'CARAVAN']

    cat_cols = ['MOSTYPE', 'MGEMLEEF', 'MOSHOOFD', 'MKOOPKLA', 'PWAPART',
                'PWABEDR', 'PWALAND', 'PPERSAUT', 'PBESAUT', 'PMOTSCO',
                'PVRAAUT', 'PAANHANG', 'PTRACTOR', 'PWERKT', 'PBROM',
                'PLEVEN', 'PPERSONG', 'PGEZONG', 'PWAOREG', 'PBRAND',
                'PZEILPL', 'PPLEZIER', 'PFIETS', 'PINBOED', 'PBYSTAND', 'AWAPART']
    num_cols = ['MAANTHUI', 'MGEMOMV', 'MGODRK', 'MGODPR', 'MGODOV', 'MGODGE', 'MRELGE',
                'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG', 'MOPLMIDD',
                'MOPLLAAG', 'MBERHOOG', 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG', 'MBERARBO',
                'MSKA', 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1', 'MAUT2',
                'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575', 'MINK7512', 'MINK123M',
                'MINKGEM', 'AWABEDR', 'AWALAND', 'APERSAUT', 'ABESAUT', 'AMOTSCO', 'AVRAAUT', 'AAANHANG',
                'ATRACTOR', 'AWERKT', 'ABROM', 'ALEVEN', 'APERSONG', 'AGEZONG', 'AWAOREG', 'ABRAND',
                'AZEILPL', 'APLEZIER', 'AFIETS', 'AINBOED', 'ABYSTAND']
    target_col = 'CARAVAN'

    df = pd.read_csv(path, sep='\t', header=None,
                     names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    extra_path_X = 'Datasets/Raw/UCI_coil2k_insurance_data_set/ticeval2000.txt'
    extra_df = pd.read_csv(extra_path_X, sep='\t', header=None,
                           names=col_names[:-1],
                           dtype={col: 'category' for col in cat_cols})
    extra_path_y = 'Datasets/Raw/UCI_coil2k_insurance_data_set/tictgts2000.txt'
    extra_df[target_col] = pd.read_csv(extra_path_y, sep='\t', header=None)

    df = df.append(extra_df).reset_index(drop=True)

    df[cat_cols] = df[cat_cols].astype('category')

    return df, cat_cols, num_cols, target_col


# #### adult
# https://archive.ics.uci.edu/ml/datasets/adult


def load_adult():
    path = 'Datasets/Raw/UCI_adult_data_set/adult.data'

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship',
                 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                 'native-country', 'target']

    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    target_col = 'target'

    df = pd.read_csv(path, sep=',', index_col=None, names=col_names,
                     dtype={col: 'category' for col in cat_cols})

    df[target_col] = np.where(df[target_col] == ' >50K', 1, 0)

    return df, cat_cols, num_cols, target_col


# #### forest
# https://archive.ics.uci.edu/ml/datasets/covertype
# Work in progress / unfinished
def load_forest():
    path = 'Datasets/Raw/UCI_forest_covertype_data_set/covtype.data'

    #     col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
    #                  'marital-status', 'occupation', 'relationship',
    #                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
    #                  'native-country', 'target']

    #     cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
    #                 'relationship', 'race', 'sex', 'native-country']
    #     num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    #     target_col = 'target'

    df = pd.read_csv(path, sep=',', index_col=None)  # , names=col_names,
    #                      dtype={col: 'category' for col in cat_cols})

    #     df[target_col] = np.where(df[target_col]== ' >50K', 1, 0)

    return df  # , cat_cols, num_cols, target_col


### helpers.py

def make_scores_dict(X_train, y_train, X_test, y_test, clf_list, print_scores=True, bootstrap=True) -> dict:
    metrics_to_use = ['auc', 'brier', 'f1', 'aps', 'acc', 'gmean', 'bacc']

    scores_dict = {'imb': y_train.mean()}

    for name, clf in clf_list:

        clf.fit(X_train, y_train.reshape(-1))
        try:
            preds = clf.predict_proba(X_test)[:, 1]
        except IndexError:
            logging.warning(f'Preds from {name} failed, only one class present.')
            preds = np.array([0] * X_test.shape[0])

        if name == 'rfc':
            if print_scores:
                print(f'Imb: {scores_dict["imb"]:.3f} ', end='')
            _scores = get_scores(y_true=y_test, y_pred=preds, metrics_to_use=metrics_to_use,
                                 bootstrap=bootstrap, print_scores=print_scores)
            scores_dict.update({metric: _scores[metric] for metric in _scores.keys()})
        else:
            _scores = get_scores(y_true=y_test, y_pred=preds, metrics_to_use=metrics_to_use,
                                 bootstrap=bootstrap, print_scores=False)

        scores_dict.update({metric + f'_{name}': _scores[metric] for metric in _scores.keys()})

    # if print_scores:
    #     print(f'Imb: {y_train.mean():.3f} ', end='')
    # # RFC
    # rfc.fit(X_train, y_train.reshape(-1))
    # try:
    #     preds_rfc = rfc.predict_proba(X_test)[:, 1]
    # except IndexError:
    #     logging.warning('Preds from RandomForest failed, only one class present.')
    #     preds_rfc = np.array([0] * X_test.shape[0])
    # scores_dict = get_scores(y_true=y_test, y_pred=preds_rfc, metrics_to_use=metrics_to_use,
    #                          bootstrap=bootstrap, print_scores=print_scores)
    # # LOGIT
    # logit.fit(X_train, y_train.reshape(-1))
    # try:
    #     preds_logit = logit.predict_proba(X_test)[:, 1]
    # except IndexError:
    #     logging.warning('Logit also failed.')
    #     preds_logit = np.array([0] * X_test.shape[0])
    # scores_dict_logit = get_scores(y_true=y_test, y_pred=preds_logit, metrics_to_use=metrics_to_use,
    #                                bootstrap=bootstrap, print_scores=False)
    #
    # scores_dict = {metric: scores_dict[metric] for metric in scores_dict.keys()}
    # scores_dict.update({metric + '_logit': scores_dict_logit[metric] for metric in scores_dict_logit.keys()})
    # scores_dict.update({'imb': y_train.mean()})

    return scores_dict


def generate_date_prefix(random_letters=True) -> str:
    out = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}-{datetime.datetime.now().minute}'
    if random_letters:
        out = f'{out}_{"".join(random.choices(string.ascii_uppercase, k=4))}'
    return out


def get_scores(y_true: np.array, y_pred: np.array,
               metrics_to_use=None,
               threshold: float = 0.5,
               bootstrap: bool = False,
               print_scores: bool = False) -> dict:
    """

    Parameters
    ----------
    y_true
    y_pred
    threshold: float
        threshold for turning scores into class labels
    print_scores: bool
        whether to print the scores

    Returns
    -------
    scores: dict
        dict of metric name: value
    """
    scores = {}
    if metrics_to_use is None:
        metrics_to_use = ['auc', 'aps', 'acc', 'f1', 'bacc', 'brier', 'gmean']

    if not bootstrap:
        # metrics that take predicted probabilities
        for name, metric in [('auc', roc_auc_score), ('aps', average_precision_score), ('brier', brier_score_loss)]:
            if name in metrics_to_use:
                scores[name] = metric(y_true, y_pred)

        # metrics that take class labels
        y_pred_bin = np.where(y_pred > threshold, 1, 0)
        for name, metric in [('acc', accuracy_score), ('bacc', balanced_accuracy_score), ('gmean', geometric_mean_score)]:
            if name in metrics_to_use:
                scores[name] = metric(y_true, y_pred_bin)
        scores['f1'] = f1_score(y_true, y_pred_bin, zero_division=0)
    else:
        scores_lists = {metric: [] for metric in metrics_to_use}
        for i in range(100):
            y_true_res, y_pred_res = resample(y_true, y_pred)
            # recursively get scores
            _scores = get_scores(y_true=y_true_res, y_pred=y_pred_res, metrics_to_use=metrics_to_use)
            for metric in metrics_to_use:
                scores_lists[metric].append(_scores[metric])

        scores = {metric: np.mean(scores_lists[metric]) for metric in metrics_to_use}

    if print_scores:
        for metric in scores.keys():
            print(f'{metric.upper()}:{scores[metric]:.4f} ', end='')
        print()

    return scores


def score_oversampling_performance(X_y_real: torch.Tensor, X_y_fake: torch.Tensor, y_real=None, y_fake=None,
                                   classifier: str = 'rfc') -> dict:
    if y_real is None:
        # assume we are in uncoditional mode and the last two columns of X_y are y as onehot ([0,1] or [1,0])
        # TODO write a better catch (for the opposite case too) or remove
        assert y_fake is None, 'score_oversampling_performance got y_real but no y_fake. Provide neither or both.'
        X_fake = X_y_fake[:, :-2]
        y_fake = X_y_fake[:, -1]
        y_fake = np.where(y_fake > 0.5, 1, 0)
        X_real = X_y_real[:, :-2]
        y_real = X_y_real[:, -1]
    else:
        # assume we are in conditional mode
        X_fake = X_y_fake
        X_real = X_y_real
        y_real = torch.Tensor(y_real).view(-1) if not isinstance(y_real, torch.Tensor) else y_real.view(-1)
        y_fake = torch.Tensor(y_fake).view(-1) if not isinstance(y_fake, torch.Tensor) else y_fake.view(-1)

    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.1, stratify=y_real)

    # only fake minority data added is class 1
    comb_X = np.vstack([X_train, X_fake[y_fake == 1]])
    comb_y = np.hstack([y_train, y_fake[y_fake == 1]])
    rfc = RandomForestClassifier(n_jobs=6, min_samples_split=10, max_depth=12)
    rfc.fit(comb_X, comb_y)
    comb_preds = rfc.predict_proba(X_test)[:, 1]
    comb_scores = get_scores(y_test, comb_preds)

    # training on only fake data
    rfc = RandomForestClassifier(n_jobs=6, min_samples_split=10, max_depth=12)
    rfc.fit(X_fake, y_fake)
    try:
        fakeonly_preds = rfc.predict_proba(X_test)[:, 1]
    except IndexError:
        logging.warning('Fakeonly preds failed, only one class present.')
        fakeonly_preds = np.array([0] * X_test.shape[0])
    fakeonly_scores = get_scores(y_test, fakeonly_preds)

    return comb_scores, fakeonly_scores


def score_real_fake(X_real: np.array, X_fake: np.array,
                    classifier: str = 'rfc') -> dict:
    rfX = np.vstack([X_real, X_fake])
    rfy = np.hstack([[1] * X_real.shape[0], [0] * X_fake.shape[0]])

    rfX_train, rfX_test, rfy_train, rfy_test = train_test_split(rfX, rfy, test_size=0.2, stratify=rfy)

    if classifier == 'logit':
        clf = LogisticRegression(max_iter=1e4, n_jobs=6)
    elif classifier == 'rfc':
        clf = RandomForestClassifier(max_depth=6, n_estimators=16, min_samples_split=100, n_jobs=6)
    elif classifier == 'rfc_shallow':
        clf = RandomForestClassifier(max_depth=2, n_estimators=16,
                                     min_samples_split=100, max_features=0.1,
                                     n_jobs=6)
    else:
        raise ValueError(f'Unknown classifier "{classifier}". Try one of "logit", "rfc", "rfc_shallow".')

    clf.fit(rfX_train, rfy_train)
    preds = clf.predict_proba(rfX_test)[:, 1]

    scores = get_scores(rfy_test, preds, print_scores=False)

    return scores


def get_dimwise_prob_metrics(X_real: np.array, X_fake: np.array,
                             y_real: np.array = None, y_fake: np.array = None,
                             measure='mean', n_num_cols: int = 0):
    if measure in ['mean', 'avg']:
        real = X_real.mean(axis=0)
        fake = X_fake.mean(axis=0)
    elif measure == 'std':
        real = X_real.std(axis=0)
        fake = X_fake.std(axis=0)
    else:
        raise ValueError(f'"measure" must be "mean" or "std" but "{measure}" was specified.')

    corr_value = pearsonr(real, fake)[0]
    rmse_value = np.sqrt(mean_squared_error(real, fake))

    if n_num_cols > 0:
        num_corr_value = pearsonr(real[:n_num_cols], fake[:n_num_cols])[0]
        num_rmse_value = np.sqrt(mean_squared_error(real[:n_num_cols], fake[:n_num_cols]))
    else:
        num_rmse_value, num_corr_value = -1, -1

    if X_real.shape[1] - n_num_cols > 0:
        cat_corr_value = pearsonr(real[n_num_cols:], fake[n_num_cols:])[0]
        cat_rmse_value = np.sqrt(mean_squared_error(real[n_num_cols:], fake[n_num_cols:]))
    else:
        cat_rmse_value, cat_corr_value = -1, -1,
    return rmse_value, corr_value, num_rmse_value, num_corr_value, cat_rmse_value, cat_corr_value


def make_num_dist_plots(X_real: np.array, X_fake: np.array,
                        y_real: np.array = None, y_fake: np.array = None,
                        show: bool = True, shape: tuple = None, subsample: bool = True,
                        num_cols=None):
    """
    Takes two arrays and plots dimension-wise kdeplots for both arrays.
    Parameters
    ----------
    X_real
    X_fake
    y_real
    y_fake
    show
    shape
    subsample

    Returns
    -------

    """
    if shape is None:
        # by default, we plot 3 columns with up to 2 rows
        if num_cols is not None:
            rows = np.minimum(len(num_cols) // 3, 2)
        else:
            rows = np.minimum(X_real.shape[1] // 3, 2)

        if rows == 0:
            shape = (1, 1)
        else:
            shape = (rows, 3)

    if subsample:
        real_size = int(np.minimum(X_real.shape[0], 5e4))
        fake_size = int(np.minimum(X_fake.shape[0], 5e4))
    else:
        real_size = X_real.shape[0]
        fake_size = X_fake.shape[0]

    fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1])
    fig.set_size_inches((8, 2.25 * shape[0]))
    # print(fig.get_figwidth(),'< width || height >', fig.get_figheight())

    for idx, ax in enumerate(axes.flatten()):
        sns.kdeplot(X_real[:real_size, idx], label='real', ax=ax, shade=True, legend=False, bw=0.02)
        sns.kdeplot(X_fake[:fake_size, idx], label='fake', ax=ax, shade=True, legend=False, bw=0.02)
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        if num_cols is not None:
            ax.set_xlabel(num_cols[idx], labelpad=-10)
    axes.flatten()[0].legend()
    plt.tight_layout()

    if show:
        plt.show()


def make_cat_dist_plots(X_real: np.array, X_fake: np.array,
                        ohe,
                        num_cols: list, cat_cols: list,
                        y_real: np.array = None, y_fake: np.array = None,
                        show: bool = True, shape: tuple = None,
                        log_counts:bool=True):
    if shape is None:
        if len(cat_cols) == 8:
            shape = (2, 4)
        elif len(cat_cols) >= 6:
            shape = (2, 3)
        elif len(cat_cols) >= 3:
            shape = (1, 3)
        elif len(cat_cols) == 2:
            shape = (1, 2)
        else:
            shape = (1, 1)
    end_idx = sum([len(c) for c in ohe.categories_]) + len(num_cols)
    X_fake_cat = pd.DataFrame(ohe.inverse_transform(X_fake[:5000, len(num_cols):end_idx]))
    X_fake_cat['type'] = 'fake'
    X_real_cat = pd.DataFrame(ohe.inverse_transform(X_real[:5000, len(num_cols):end_idx]))
    X_real_cat['type'] = 'real'
    X_real_fake_cat = pd.concat([X_real_cat, X_fake_cat])
    X_real_fake_cat.columns = cat_cols + ['type']

    fig, axes = plt.subplots(shape[0], shape[1])
    fig.set_size_inches((4 * shape[0], 1.12 * shape[1]))
    # print(fig.get_figwidth(),'< width || height >', fig.get_figheight())

    for idx, ax in enumerate(axes.flatten()):
        _plot = sns.countplot(x=cat_cols[idx], hue='type',
                              data=X_real_fake_cat, ax=ax,
                              order=X_real_cat.iloc[:, idx].value_counts().index)
        if idx > 0:
            ax.get_legend().remove()
        else:
            ax.get_legend().remove()
            ax.legend(loc=1)
            ax.get_legend().set_title(None)
        if log_counts:
           _plot.set_yscale("log")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.set_ylabel(None)
    plt.tight_layout()
    if show:
        plt.show()


def make_dimwise_probability_plot(X_real: np.array, X_fake: np.array,
                                  y_real: np.array = None, y_fake: np.array = None,
                                  measure='mean',
                                  show=True, make_fig=True, ax=None,
                                  show_rmse=True, show_corr=True) -> Tuple[float, float]:
    """
    Takes two arrays and plots a scatter plot of a measure (i.e.) the mean for each column of the two arrays against
    each other. The name comes from Bernoulli success probabilities for binary variables, i.e. their mean,
    but this approach generalises to numerical columns. All variables are assumed to be scaled to [0,1].

    Note: Since metrics are computed column-wise, a onehot-encoded column of k-cardinality has k times the impact of a
    single numerical column. Thus it might be wise to compute metrics for both kinds of columns separately.

    Reference: Choi et al., 2017
    Parameters
    ----------
    X_real: np.array
        Array of real data
    X_fake: np.array
        Array of synthetic data
    y_real
    y_fake
    measure: str
        Which measure to plot. Options are ['mean', 'std'].
    show: bool
        Whether to call plt.show()
    make_fig: bool
        Whether to create a new plt figure
    ax:
        plt axes object to plot on
    show_rmse: bool
        Whether to add rmse to the plot
    show_corr: bool
        Whether to add pearson corr coeff to the plot

    Returns
    -------
    rmse_value: float
        root mean square error between the vectors of dimension-wise measure for both arrays.
    corr_value: float
        pearson correlation coefficient between the vectors of dimension-wise measure for both arrays.
    """

    if make_fig and ax is None:
        fig, ax = plt.subplots(1)

    if measure in ['mean', 'avg']:
        real = X_real.mean(axis=0)
        fake = X_fake.mean(axis=0)
    elif measure == 'std':
        real = X_real.std(axis=0)
        fake = X_fake.std(axis=0)
    else:
        raise ValueError(f'"measure" must be "mean" or "std" but "{measure}" was specified.')

    upper_bound = np.maximum(np.max(real) * 1.1, np.max(fake) * 1.1)
    upper_bound = np.minimum(1, upper_bound)

    if measure in ['mean', 'avg']:
        upper_bound = 1
    else:
        upper_bound = 0.6

    ax.scatter(x=real, y=fake)
    ax.plot([0, 1, 2], linestyle='--', c='black')
    ax.set_xlabel('Real')
    ax.set_ylabel('Fake')
    ax.set_xlim(left=0, right=upper_bound)
    ax.set_ylim(bottom=0, top=upper_bound)

    corr_value = pearsonr(real, fake)[0]
    rmse_value = np.sqrt(mean_squared_error(real, fake))

    s = ""
    if show_rmse:
        s += f'RMSE: {rmse_value:.4f}\n'
    if show_corr:
        s += f'CORR: {corr_value:.4f}\n'
    if s != "":
        ax.text(x=upper_bound * 0.98, y=0,
                s=s,
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom')

    if show:
        plt.show()

    return rmse_value, corr_value


def make_dimwise_prediction_performance_plot(X_real: np.array, X_fake: np.array,
                                             y_real: np.array = None, y_fake: np.array = None,
                                             n_dims_to_plot: int = 0,
                                             n_num_cols: int = None,
                                             cat_input_dims: list = None,
                                             show=True, make_fig=True, ax=None,
                                             show_rmse=True, show_corr=True) -> Tuple[float, float]:
    """

    Parameters
    ----------
    X_real: np.array
        Array of real data
    X_fake: np.array
        Array of synthetic data
    y_real
    y_fake
    n_num_cols: int
        number of numerical columns, assumed to be come first

    n_dims_to_plot: int
        the first 'n_dims_to_plot' columns will be plotted. All columns will be used for model fitting
    show: bool
        Whether to call plt.show()
    make_fig: bool
        Whether to create a new plt figure
    ax:
        plt axes object to plot on
    show_rmse: bool
        Whether to add rmse to the plot
    show_corr: bool
        Whether to add pearson corr coeff to the plot

    Returns
    -------
    rmse_value: float
        root mean square error between the vectors of dimension-wise measure for both arrays.
    corr_value: float
        pearson correlation coefficient between the vectors of dimension-wise measure for both arrays.

    """
    # TODO allow to pass saved values for X_real, since we only need to compute it once during training

    if make_fig and ax is None:
        fig, ax = plt.subplots(1)

    if n_num_cols is None:
        n_num_cols = X_real.shape[1]
    if n_dims_to_plot == 0:
        n_dims_to_plot = n_num_cols if cat_input_dims is None else n_num_cols + len(cat_input_dims)

    real, fake = [], []
    for idx in range(n_dims_to_plot):
        for results_list, arr in [(real, X_real), (fake, X_fake)]:
            # Linear regression when using numerical columns as target
            # we use ridge to lessen the need for preprocessing
            if idx < n_num_cols:
                X = arr.copy()[:, ~np.eye(X_real.shape[1])[idx].astype(bool)]
                y = arr.copy()[:, [idx]]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = Ridge()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results_list.append(r2_score(y_test, preds))
            # RandomForest when using categorical columns as target
            else:
                # get cardinality of target
                n_classes = cat_input_dims[idx - n_num_cols]
                start_idx = n_num_cols + sum(cat_input_dims[:idx - n_num_cols])
                end_idx = start_idx + n_classes
                X = np.delete(arr, np.arange(start_idx, end_idx), axis=1)
                y = arr.copy()[:, start_idx: end_idx]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = RandomForestClassifier(n_estimators=20, min_samples_split=0.1, max_depth=6, n_jobs=6)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results_list.append(f1_score(y_test, preds, average='weighted', zero_division=0))

    upper_bound = np.maximum(np.max(real) * 1.1, np.max(fake) * 1.1)
    upper_bound = np.minimum(1, upper_bound)

    ax.scatter(x=real, y=fake)
    ax.plot([0, 1], linestyle='--', c='black')
    ax.set_xlabel('Real')
    ax.set_ylabel('Fake')
    ax.set_xlim(left=0, right=upper_bound)
    ax.set_ylim(bottom=0, top=upper_bound)

    corr_value = pearsonr(real, fake)[0]
    rmse_value = np.sqrt(mean_squared_error(real, fake))

    s = ""
    if show_rmse:
        s += f'RMSE: {rmse_value:.4f}\n'
    if show_corr:
        s += f'CORR: {corr_value:.4f}\n'
    if s != "":
        ax.text(x=upper_bound * 0.98, y=0,
                s=s,
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom')

    if show:
        plt.show()

    return rmse_value, corr_value


def save_current_plot(name: str, prefix: str = '', path='', show=False, clear=False):
    filename = f'{path}/{prefix}{name}.pdf'
    plt.savefig(filename, dpi=100)
    if show:
        # TODO clean up. hacky solution to surpress plots when not developing
        # plt.clf()
        # plt.close()
        plt.show()
    if clear:
        plt.clf()


def get_cat_dims(X, cat_cols) -> list:
    """
    Takes a pd.DataFrame and a list of columns and returns a list of levels/cardinality per column in the same order.
    """
    return [(X[col].nunique()) for col in cat_cols]
