from pandas import DataFrame as df
import pandas as pd

from analysis.data_processing import *
from analysis.visualization import *
from settings import *


is_plot=False

data_file_name='question_session_log_13.csv'
event_data_file_name='temp_events_table_670000' #'events_table_full.csv'
event_data=df.from_csv(os.path.join(DATA_ROOT,event_data_file_name))

"""event_data_head=event_data.tail(1000)
event_data_head.to_csv(os.path.join(DATA_ROOT,'temp_event_data_head100.csv'))"""
raw_data=df.from_csv(os.path.join(DATA_ROOT,data_file_name))
'''columns  = ['student_id', 'lo_id', 'num_of_questions_in_lo_session', 'question_id',
       'domain_id', 'question_start_time', 'question_end_time',
       'number_of_attempts', 'used_hists_or_assistence',
       'is_correct_at_first_attempt', 'first_attemp_time', 'success_time',
       'bestScore', 'action_count', 'action_hist', 'session_count',
       'last_response', 'q_f_type', 'q_f_subjects', 'q_f_detail', 'q_f_goals',
       'q_f_representation', 'q_f_activity_type',
       'adjusted_is_correct_at_first_attempt',
       'num_of_full_attempts_to_success', 'weighted_score', 'smart_start_time',
       'raw_duration', 'duration_in_seconds', 'time_to_first_attemp_in_sec',
       'time_to_success', 'is_success']'''
lo_index=df.from_csv(os.path.join(DATA_ROOT,'LOs_order_index_fraction.csv'))
'''Index(['sTreeTitle', 'sLanguage', 'sLOTitle', 'LO_subject_index',
       'LO_subsubject_index', 'LO_general_index', 'LO_combined_index'],
      dtype='object')'''

meta_data=df.from_csv(os.path.join(DATA_ROOT,'MD_fractions.csv'))
'''Index(['sElementID', 'gLO', 'nVersion', 'sName', 'nQuestionIndex',
       'sQuestionPageID', 'nLanguage', 'dtCreatedDate', 'sSyllabus', 'sLOurl',
       'nPages', 'sQtype', 'sSubjects', 'sSubSubjects', 'sDetail', 'sGoals',
       'sRepresentation', 'sActivityType'],'''


#data=data.head(5000)

# calculate basic success measures
success_direct_measures_list=['num_of_events','num_of_successes','num_of_correct_at_first_attemps','is_success','adjusted_is_correct_at_first_attempt']
success_advanced_measures_list=['is_success','time_to_success','num_of_full_attempts_to_success','adjusted_is_correct_at_first_attempt']
index_list=['LO_subject_index','LO_subsubject_index']
meta_data_list=['q_f_type', 'q_f_subjects', 'q_f_detail', 'q_f_goals','q_f_representation', 'q_f_activity_type']

data=preprocess_data(raw_data, meta_data, lo_index)

agg_data=aggregate_and_sort_data(data)

if is_plot:
    simple_df_plot(agg_data, index_list+success_direct_measures_list,save_name='adj_direct_success_measures.png', OVERRIDE=True )
    simple_df_plot(agg_data, index_list+success_advanced_measures_list, save_name='adj_advance_success_measures.png', OVERRIDE=True )
    simple_df_plot(agg_data, ['is_success','adjusted_is_correct_at_first_attempt'],
                   save_name='success_vs_correct_at_first_attempt.png', is_subplots=False, figsize=(20,3), OVERRIDE=True )

simple_df_plot(agg_data, ['LO_subsubject_index','adjusted_is_correct_at_first_attempt'],
                   save_name='LO_subsubject_vs_correct_at_first_attempt.png',secondary_y=True, is_subplots=False, figsize=(20,3), OVERRIDE=True )


# calculate advanced success measures - only on successed questions

for md_col in meta_data_list+index_list:
    boxplot(agg_data, column='adjusted_is_correct_at_first_attempt', by_col=md_col , save_name='',OVERRIDE=True)

data_filtered_by_success=data.loc[data['is_success'] == 1]

success_advanced_measures=data_filtered_by_success.groupby('question_id').mean()[index_list+success_measures_list].copy()
success_advanced_measures.sort_values(['question_ind'], inplace=True)
success_advanced_measures.reset_index(inplace=True)
success_advanced_measures.index.name='question_ind'
success_advanced_measures.drop(['question_ind'], axis=1, inplace=True)

success_advanced_measures.plot(subplots=True, figsize=(20,15))

plt.savefig(os.path.join(OUTPUT_DIR,'mean_success_advanced_measures.png'))
plt.close()
#data_filtered_by_failed
print(data.head())

print('')