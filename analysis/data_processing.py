from pandas import DataFrame as df
import pandas as pd
import os
from settings import *

def preprocess_data(raw_data, meta_data, lo_index, is_fractions_only=True, is_add_lo_indeces=True,is_remove_embbeded_assets_questions=True):
    if is_fractions_only:
        data = raw_data.loc[raw_data.domain_id == 'frozenset({1})'].copy()
    else:
        data = raw_data

    # data=data.head(1000)
    data.reset_index(inplace=True, drop=True)
    data['is_success'] = (data['time_to_success'] > 0).apply(float)

    if is_remove_embbeded_assets_questions:
        # filter out embbede assets question without time to success (caused by a bug)
        EmbeddedAsset_questions = meta_data.loc[meta_data.sQtype == 'EmbeddedAsset']['sElementID'].values
        data['is_embbeded_assets'] = [True if q in EmbeddedAsset_questions else False for q in data.question_id]

        embedded_asset_data = data.loc[data.is_embbeded_assets == True]

        filter_out_embbeded_assets_questions = embedded_asset_data.time_to_success.dropna().index
        adjusted_data_index = [d for d in data.index if d not in filter_out_embbeded_assets_questions]
        data = data.loc[adjusted_data_index].reset_index(drop=True)

    if is_add_lo_indeces:
        lo_index.index = [s.lower() for s in lo_index.index]
        # arrange indeces
        data_lo_index_details = lo_index.loc[data.lo_id].reset_index(drop=True)
        data[data_lo_index_details.columns] = data_lo_index_details
        data['question_ind'] = data.LO_general_index + 0.01 * data.num_of_questions_in_lo_session

    processed_data=data

    return processed_data

def aggregate_and_sort_data(processed_data, agg_col='question_id',sort_by_col='question_ind', ascending=True, is_drop_sort_col=True,filter_dict=["['is_success']>0"]):
    agg_data=processed_data.groupby(agg_col).mean()
    agg_data['num_of_successes']=processed_data.groupby(agg_col).sum()['is_success']
    agg_data['num_of_correct_at_first_attemps'] = processed_data.groupby(agg_col).sum()['adjusted_is_correct_at_first_attempt']
    agg_data['num_of_events']=processed_data.groupby(agg_col).count()['is_success']

    agg_q_f_data=processed_data.groupby(agg_col)['q_f_type', 'q_f_subjects', 'q_f_detail', 'q_f_goals',
    'q_f_representation', 'q_f_activity_type'].apply(lambda s:s.iloc[0])

    agg_q_f_data=agg_q_f_data.applymap(lambda s: s[s.find("'")+1:s.rfind("'")])
    agg_q_f_data=agg_q_f_data.applymap(lambda s: s if s[1].lower().islower() else s[::-1]) #reverse hebrew words

    agg_data=pd.concat([agg_data,agg_q_f_data],axis=1)

    for filter in filter_dict:
        agg_data=agg_data.loc[eval('agg_data'+filter)] #filter out questions where no student answered correctly (probably a bug)

    agg_data=agg_data.sort_values([sort_by_col], ascending=ascending).copy() #sort by question index
    agg_data.reset_index(inplace=True)
    agg_data.index.name = sort_by_col
    if is_drop_sort_col:
        agg_data=agg_data.drop([sort_by_col], axis=1).copy()

    return agg_data


def add_LO_indexes_to_meta_data(meta_data=None, lo_index=None, is_load_data=True, md_file='MD_math_processed.csv', loi_file='LOs_order_index_fraction.csv', filter_by_language=True, save_to_csv=True, save_name='MD_math_processed.csv'):
    if is_load_data:
        meta_data= df.from_csv(os.path.join(DATA_ROOT,md_file))
        lo_index = df.from_csv(os.path.join(DATA_ROOT,loi_file))
    # lo_index.index = [s.lower() for s in lo_index.index]
    meta_data = meta_data.loc[meta_data.nLanguage == 1]
    meta_data = meta_data.reset_index(drop=True)
    lo_index_ordered = lo_index.loc[meta_data.gLO].reset_index(drop=True)
    meta_data[lo_index_ordered.columns] = lo_index_ordered
    if 'num_of_questions_in_lo_session' in meta_data.columns:
        meta_data.num_of_questions_in_lo_session=meta_data['sQuestionPageID'].apply(lambda s: int(s[s.rfind('_') + 1:]))

    meta_data['question_index'] = meta_data.LO_general_index + 0.01 * meta_data.num_of_questions_in_lo_session

    if filter_by_language:
        meta_data = meta_data.loc[meta_data.nLanguage == 1]

    meta_data.drop_duplicates(inplace=True)
    if save_to_csv:
        meta_data.to_csv(os.path.join(DATA_ROOT,save_name))
    return meta_data