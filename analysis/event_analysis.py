from pandas import DataFrame as df
import pandas as pd
import numpy as np

from analysis.data_processing import *
from analysis.visualization import *
from settings import *
from scipy.stats import entropy
from scipy.stats import chi2_contingency


class EventsAnalyzer():
    def __init__(self):
        self.event_data = None
        self.meta_data = None
        self.processed_event_data = None
        self.top_questions = None
        return

    def load_data(self, agg_data_file_name='question_session_log_13.csv',
                      event_data_file_name='events_table_full.csv', meta_data_file_name='MD_math_processed.csv'):
        #event_data_file_name = 'events_table_full.csv'
        print('loading event data from %s...' %os.path.join(DATA_ROOT,event_data_file_name))
        self.event_data=df.from_csv(os.path.join(DATA_ROOT,event_data_file_name))
        print(self.event_data.columns)
        '''Index(['event_id', 'question_id', 'time', 'student_id', 'object_url', 'lo_id',
               'session_id', 'page_url', 'action', 'completion', 'score', 'response',
               'clean_response', 'full_or_partial_success', 'full_attempts',
               'full_attempts_desc', 'full_answer', 'n_sections'],
              dtype='object')'''

        print('loading event data from %s...' % os.path.join(DATA_ROOT, meta_data_file_name))
        self.meta_data=df.from_csv(os.path.join(DATA_ROOT,meta_data_file_name))
        '''Index(['sElementID', 'gLO', 'nVersion', 'sName', 'nQuestionIndex',
               'sQuestionPageID', 'nLanguage', 'dtCreatedDate', 'sSyllabus', 'sLOurl',
               'nPages', 'sQtype', 'sSubjects', 'sSubSubjects', 'sDetail', 'sGoals',
               'sRepresentation', 'sActivityType'],'''

        '''data.reset_index(inplace=True)
        lo_index = df.from_csv(os.path.join(DATA_ROOT, 'LOs_order_index_fraction.csv'))
        lo_index.index = [s.lower() for s in lo_index.index]
        # add LO index to data :
        data_lo_index_ordered = lo_index.loc[data.lo_id].reset_index()[['index', 'LO_subject_index',
                                                                        'LO_subsubject_index', 'LO_general_index',
                                                                        'LO_combined_index']]
        data[data_lo_index_ordered.columns] = data_lo_index_ordered

        if 'num_of_questions_in_lo_session' in self.meta_data.columns:
            question_index = self.meta_data.num_of_questions_in_lo_session
            question_index.drop_duplicates(inplace=True)
        else:
            question_index = self.meta_data['sQuestionPageID'].apply(lambda s: int(s[s.rfind('_') + 1:]))
            question_index.name = 'num_of_questions_in_lo_session'
        question_index.index = self.meta_data.sElementID
        question_index_ordered = question_index.loc[data.question_id].reset_index()
        data['num_of_questions_in_lo_session'] = question_index_ordered['num_of_questions_in_lo_session']

        # add question - n_in_session
        data['question_general_ind'] = data.LO_general_index + 0.01 * data.num_of_questions_in_lo_session
        print(data.loc[data.question_id=='question_4e1d69bf-7016-4dfc-a9ab-d7459e250b1b'])
        data.sort_values('question_general_ind', inplace=True)
        questions_order = data['question_general_ind']
        questions_order.index = data['question_id']
        print(len(questions_order))
        questions_order.drop_duplicates(inplace=True)
        print(len(questions_order))
        print(len(set(questions_order.index)))'''

    def preprocess_event_data(self,filter_by_meta_data=True, filter_only_single_section_data=True, filter_only_first_attempt=True, filter_by_min_n_students=False, filter_by_min_n_answers=False, filter_by_language=True, arrange_by_index=True):
        """
        Adds 'n_attempt' field to data (number of attempt at a specific question for each student)
          filter data by 'asked_check' and by params.
        :param
        filter_only_single_section_data: if True -- use_only questions with single section
         filter_only_first_attempt: if True -- use only the first attempt of the student in each question
        filter_by_min_n_students: use only questions answered by min num of students - optional
        filter_by_min_n_answers: se only questions which had min number of unique responses - optional

        :return:
        processed and filtered data
        """
        if type(self.event_data)!= pd.core.frame.DataFrame:
            self.load_data()


        data=self.event_data.loc[self.event_data.action=='asked_check']
        print('filtering and processing data...(N=%i)' % len(data))

        if filter_by_meta_data:
            questions_list_md=set(data['question_id']).intersection(set(self.meta_data.sElementID))
            data_in_md_index=[i for i in data.index if data.question_id.loc[i] in questions_list_md]
            data=data.loc[data_in_md_index]
            print('filtering only questions in meta data...(N=%i)' % len(data))

        if filter_only_single_section_data:
            if 'n_sections' not in data.columns:
                data_with_response=data['full_attempts_desc'].dropna()
                n_sections=data_with_response.apply(lambda s: len(eval(s)))
                data=pd.concat([data, n_sections],axis=0)
            data = data.loc[data.n_sections == 1]

        if filter_by_language:
            filtered_questions=self.meta_data.loc[self.meta_data.nLanguage==1].sElementID.values
            filtered_index = [i for i in data.index if data.question_id[i] in filtered_questions]
            data = data.loc[filtered_index]
            print('filtered nLanguage=1 only... (N=%i)' %len(data))

        data['n_attempt'] = data.groupby(['question_id','student_id']).full_attempts.cumsum()

        if filter_only_first_attempt:
            print('filter first attempt only...(N=%i)' %len(data))
            data=data.loc[data.n_attempt==1]

        if type(filter_by_min_n_students)==int:
            n_students_per_question=data.groupby('question_id')['student_id'].value_counts().groupby('question_id').count()
            filtered_questions=n_students_per_question.loc[n_students_per_question >= filter_by_min_n_students].index
            filtered_index=[i for i in data.index if data.question_id[i] in filtered_questions]
            data=data.loc[filtered_index]
            print('filter n_students>%i only... (N=%i)' % (filter_by_min_n_students,len(data)) )


        if type(filter_by_min_n_answers)==int:

            n_students_per_question = data.groupby('question_id')['clean_response'].value_counts().groupby('question_id').count()
            filtered_questions = n_students_per_question.loc[n_students_per_question >= filter_by_min_n_answers].index
            filtered_index = [i for i in data.index if data.question_id[i] in filtered_questions]
            data = data.loc[filtered_index]
            print('filter n_answers>%i only... (N=%i)' %(filter_by_min_n_answers,len(data)))

        if arrange_by_index:
            data.reset_index(inplace=True,drop=True)
            question_indexes = self.meta_data[['LO_subject_index', 'LO_subsubject_index',
                                              'LO_general_index', 'LO_combined_index', 'question_index']].copy()
            question_indexes.index=self.meta_data.sElementID
            question_indexes.drop_duplicates(inplace=True)

            data[question_indexes.columns]=question_indexes.loc[data.question_id].reset_index(drop=True)

            data.sort_values(by='question_index',inplace=True)

        self.processed_event_data = data

        #arrange questions list by LO and questions indexes
        questions_list=list(set(self.processed_event_data.question_id).intersection(set(self.meta_data.sElementID )))
        questions_list_ordered = question_indexes.loc[questions_list]
        questions_list_ordered=questions_list_ordered['question_index'].sort_values().dropna().index
        self.processed_questions_list=list(questions_list_ordered)
        self.question_indexes=question_indexes
        print('number of questions = %i' % len(self.processed_questions_list))


#describe events per question to find interesting questions:
    def agg_by_question(self, is_plot=False, min_num_students=150, min_answers=3, max_answers=200, n_questions='all'):
        question_description = df()
        question_description['students_per_question'] = self.processed_event_data.groupby('question_id')['student_id'].value_counts().groupby('question_id').count()
        question_description['events_per_question'] = self.processed_event_data.groupby('question_id')['event_id'].value_counts().groupby('question_id').count()
        question_description['answers_per_question'] = self.processed_event_data.groupby('question_id')['clean_response'].value_counts().groupby('question_id').count()
        #number_of_responses_per_question=single_section_data.groupby('question_id')['clean_response'].value_counts()

        question_description.sort_values('students_per_question',ascending=False,inplace=True)

        #filter ineresting questions:

        interesting_questions= question_description.loc[question_description.students_per_question>min_num_students].loc[question_description.answers_per_question>min_answers].loc[question_description.answers_per_question<max_answers]
        interesting_questions.sort_values('answers_per_question',ascending=False,inplace=True)
        """Index(['question_25504e71-8563-4600-aa5e-be589b5e44c5',
       'question_b39b61da-cd9c-4805-81fd-df32c4affb97',
       'question_d20b14ef-b0ae-4d10-8a33-83246de7af84',
       'question_d9bc9247-5649-42d5-bc73-c6794c3cb640',
       'question_8af33bc4-0084-466e-8ab1-08448f4f277e'])"""
        if is_plot:
            simple_df_plot(question_description, title='events - question description', save_name='questions_event_description.png', reset_index=True, OVERRIDE=True)

        self.question_description = question_description
        if n_questions == 'all':
            self.top_questions = interesting_questions
        else:
            self.top_questions=interesting_questions.iloc[:min(n_questions,len(interesting_questions))]

        return self.top_questions.index

    def agg_by_question_attempts(self,only_top_questions=True,is_plot=False):
        if type(self.event_data)!= pd.core.frame.DataFrame:
            self.load_data()
        if only_top_questions:
            if not self.top_questions:
                self.agg_by_question()

                n_attempts_per_question = \
                self.processed_event_data.groupby('question_id')[
                    'n_attempt'].value_counts().unstack().loc[self.top_questions.index]
                n_attempts_per_question.index=range(len(n_attempts_per_question))
                n_attempts_per_question.index+=1

        if is_plot:
            simple_df_plot(n_attempts_per_question.T, kind='bar', is_subplots=False, is_legend=False, figsize=(15, 5),
                           xlabel='attempt', ylabel='count',
                           title='events - question attempts description',
                           save_name='questions_event_attempts_description.png', OVERRIDE=True)
            simple_df_plot(n_attempts_per_question.T.iloc[1:].T, kind='bar', is_subplots=False, is_legend=False, figsize=(15, 5),
                           xlabel='attempt (>1)', ylabel='count',
                           title='events - question attempts description',
                           save_name='questions_event_attempts_description.T.png',  OVERRIDE=True)

        return n_attempts_per_question

    def add_error_type_to_event_table(self):
        
class QuestionEventAnalyzer(EventsAnalyzer):

    def get_question_event_description(self,question_id, is_plot=False, is_clean_response=True):
        if type(self.event_data)!= pd.core.frame.DataFrame:
            self.load_data()

        question_events_raw=self.event_data.loc[self.event_data['question_id']==question_id]

        responses_hist=df(question_events_raw['clean_response'].value_counts())

        is_correct_answer = [question_events_raw.loc[question_events_raw['clean_response']==r]['score'].sum()>0 for r in responses_hist.index]
        responses_hist.index = ['%s*' %r.split("'")[3] if is_correct_answer[i] else '%s' %r.split("'")[3] for  i,r in enumerate(responses_hist.index)]

        if is_plot:
            responses_hist.plot(kind='bar')
            plt.title(question_id)
            plt.savefig(os.path.join(OUTPUT_DIR, 'responses_hist_%s.png' %question_id))
        responses_hist['is_correct_answer'] = is_correct_answer

        return responses_hist

    def get_response_vector(self, question_id, non_common_response_threshold=0.02, max_number_of_answers=10, drop_correct_answer=True, clean_response=False):
        mark_as_other=[]
        raw_responses = self.processed_event_data.loc[self.processed_event_data.question_id == question_id][
            ['student_id', 'clean_response', 'score']]
        correct_answers = set(raw_responses.loc[raw_responses.score == 1]['clean_response'])
        raw_responses.set_index('student_id', drop=True, inplace=True)
        raw_responses = raw_responses['clean_response']
        raw_responses.name = question_id
        if drop_correct_answer:
            for r in correct_answers:
                raw_responses = raw_responses[raw_responses != r]

        #mark all non frequent responses as 'other' and sum over them
        raw_marginal_frequency = raw_responses.value_counts() / len(raw_responses)
        if len(raw_marginal_frequency)>max_number_of_answers:
            mark_as_other.extend(raw_marginal_frequency.index[max_number_of_answers:]) #leave only the 10 most common answers
        if len(raw_responses)==0:
            return df(), df()
        non_common_response_threshold=max(1/len(raw_responses), non_common_response_threshold) # if only one student answered mark as 'other'
        non_frequent_responses = raw_marginal_frequency.loc[raw_marginal_frequency < non_common_response_threshold].index
        mark_as_other.extend(non_frequent_responses)
        for r in set(mark_as_other):
            raw_responses[raw_responses==r]='other'



        #set maximum


        if not drop_correct_answer: #mark correct answers with a star
            marginal_frequency.index = ['%s*' %r if r in correct_answers else r for r in marginal_frequency.index]
            raw_responses=['%s*' %r if r in correct_answers else r for r in raw_responses]


        if clean_response:
            cleaner=lambda s: s[s.find(': ')+3:-2].replace("\\\\","").replace("frac","").replace("}{","/")
            raw_responses=raw_responses.apply(cleaner)

        marginal_frequency = raw_responses.value_counts() / len(raw_responses)
        #marginal_frequency['other']=non_frequent_responses.sum()
        #marginal_frequency.drop(non_frequent_responses.index, inplace=True)
        return raw_responses, marginal_frequency

    def get_common_mistakes_df(self, questions_list, LOs='all', common_mistake_threshold=0.02, n_common_mistakes=5, save_name='temp_common_errors.csv', OVERRIDE=False):

        meta_data_columns = ['sQuestionPageID', 'sLOurl', 'LO_subject_index', 'LO_subsubject_index',
                             'num_of_questions_in_lo_session']

        md = self.meta_data[meta_data_columns].copy()
        md.index = self.meta_data.sElementID
        md = md.loc[questions_list]
        md.drop_duplicates(inplace=True)

        common_mistakes_df = df(index=pd.MultiIndex.from_product([questions_list,range(n_common_mistakes+1)]),columns=['mistake','p','n']+meta_data_columns)

        qLO_prev=0
        for question_id in questions_list:
            qLO=md.loc[question_id]['LO_subject_index']
            print('-%s - question_id' %qLO)
            if LOs!='all':
                if qLO in LOs:
                    pass
                else:
                    continue
            raw_responses, common_mistakes=self.get_response_vector(question_id, non_common_response_threshold=common_mistake_threshold, max_number_of_answers=n_common_mistakes,clean_response=True)
            '''responses_hist = self.get_question_event_description(question_id)
            correct_responses = responses_hist['clean_response'].loc[responses_hist['is_correct_answer'] == True]
            mistakes_count_hist = responses_hist['clean_response'].loc[responses_hist['is_correct_answer'] == False]

            mistakes_percent_hist = mistakes_count_hist / mistakes_count_hist.sum()

            common_mistakes = marginal_frequency.loc[mistakes_percent_hist>common_mistake_threshold]
            if len(common_mistakes)>n_common_mistakes:
                common_mistakes=common_mistakes.head(n_common_mistakes)'''
            for i,ind in enumerate(common_mistakes.index):
                common_mistakes_df.loc[question_id, i]['mistake']= ind
                common_mistakes_df.loc[question_id, i]['p'] = common_mistakes.loc[ind]
                common_mistakes_df.loc[question_id, i]['n'] = common_mistakes.loc[ind]*len(raw_responses)
                common_mistakes_df.loc[question_id, i][meta_data_columns] = md.loc[question_id]
        if qLO_prev!=qLO:
            common_mistakes_df.to_csv(os.path.join(OUTPUT_DIR, 'temp_common_mistakes_LO1-%s.csv' %qLO))
        qLO_prev = qLO
        #common_mistakes_df=common_mistakes_df.unstack()
        # add meta data

        common_mistakes.dropna(how='all',inplace=True)
        common_mistakes.reset_index(inplace=True)

        if OVERRIDE:
            common_mistakes_df.to_csv(os.path.join(OUTPUT_DIR,save_name))

        return common_mistakes_df

    def get_question_features(self, question_id, top_answers_threshold_list=[0.3,0.5,0.8], common_mistake_threshold=0.10, non_common_mistake_threshold=0.01, return_features='all', save_name='temp.png',is_plot_mistake_description=False ):

        F=dict() #question features dict
        responses_hist=self.get_question_event_description(question_id)
        correct_responses=responses_hist['clean_response'].loc[responses_hist['is_correct_answer']==True]
        mistakes_count_hist = responses_hist['clean_response'].loc[responses_hist['is_correct_answer'] == False]

        mistakes_percent_hist = mistakes_count_hist / mistakes_count_hist.sum()
        cumsum_mistakes_percent_hist = mistakes_percent_hist.cumsum()
        mistakes_desc=pd.concat([mistakes_count_hist,mistakes_percent_hist,cumsum_mistakes_percent_hist],axis=1)
        mistakes_desc.columns=['answer_count', 'answer_percent', 'percent_cumsum']

        F['n_wrong_answers']=len(mistakes_count_hist.index)
        F['n_correct_answers'] = len(correct_responses.index)
        F['percent_wrong_answers']=mistakes_count_hist.sum()/responses_hist['clean_response'].sum()
        F['mistakes_normalized_entropy'] = entropy(mistakes_percent_hist)/np.log(F['n_wrong_answers'])

        for t in top_answers_threshold_list:
            F['top%.02f' %t]=(cumsum_mistakes_percent_hist <= t).sum() + 1

        common_mistakes= mistakes_percent_hist.loc[mistakes_percent_hist >= common_mistake_threshold]
        non_common_mistakes= mistakes_percent_hist.loc[mistakes_percent_hist <= max(non_common_mistake_threshold, 1 / mistakes_count_hist.sum())]

        F['n_common_mistakes(>%.02f)' %common_mistake_threshold]=len(common_mistakes)
        F['percent_common_mistakes(>%.02f)' % common_mistake_threshold] = common_mistakes.sum()
        F['n_non_common_mistakes (<%.02f or 1 student)' %non_common_mistake_threshold] =len(non_common_mistakes)
        F['percent_non_common_mistakes (<%.02f or 1 student)' % non_common_mistake_threshold] = non_common_mistakes.sum()

        if is_plot_mistake_description:
            simple_df_plot(mistakes_desc,kind='bar',is_subplots=True,save_name='mistake_histogram_%s.png' %question_id,OVERRIDE=True, title=question_id)

        features=df.from_dict(F,orient='index')[0]
        features.name=question_id

        self.mistake_description=mistakes_desc
        if return_features=='all':
            self.question_event_features=features
            return features
        else:
            self.question_event_features = features[return_features]
            return features[return_features]

    def get_all_questions_features(self, questions=None, min_num_students=150, min_answers=5, max_answers=200 , is_plot=False ,is_plot_by_columns=False, plot_columns='all', is_save_csv=False):
        if not questions:
            self.load_data()
            top_questions = self.agg_by_question(min_num_students=min_num_students, min_answers=min_answers, max_answers=max_answers)
            questions=top_questions

        question_features = df(columns=questions)
        print('calculating featuers for top %i questions...' %len(questions))
        for question_id in questions:
            question_features[question_id]=self.get_question_features(question_id)
        question_features=question_features.T
        question_features['n_students']=self.question_description['students_per_question'].loc[questions]
        question_features.sort_values('n_students', ascending=False, inplace=True)
        question_features.sort_index(inplace=True)
        question_features['common_non_common_ratio'] = question_features['n_common_mistakes(>0.10)']/question_features['n_non_common_mistakes (<0.01 or 1 student)']
        self.question_features=question_features

        '''basic_question_features_list=['n_students','n_correct_answers','n_wrong_answers','percent_wrong_answers']
        percentile_question_features_list=['top0.30', 'top0.50', 'top0.80']
        entropy_question_features_list=['mistakes_normalized_entropy']
        advance_question_features_list=['n_common_mistakes(>0.10)','n_non_common_mistakes (<0.01 or 1 student)']
        advance_question_features_list = ['perent_common_mistakes(>0.10)', 'percent_non_common_mistakes (<0.01 or 1 student)']'''''

        if is_save_csv:
            self.question_features.to_csv(os.path.join(OUTPUT_DIR, 'Question_features.csv'))

        if is_plot and not is_plot_by_columns:
            simple_df_plot(question_features, columns=plot_columns,args={'kind':'bar', 'reset_index':True},
                           title='Events Data - question features', save_name='questions_event_features .png',  OVERRIDE=True)
        elif is_plot_by_columns:
            simple_df_plot(question_features,
                           columns=['n_students', 'n_correct_answers', 'n_wrong_answers', 'percent_wrong_answers'],
                           kind = 'bar', is_legend = False, is_subplots = True, reset_index = True,
                           title='Events Data - basic question features',
                           save_name='questions_event_features- basic.png',
                           OVERRIDE=True)
            simple_df_plot(question_features, columns=['top0.30', 'top0.50', 'top0.80'],
                           kind='bar', is_legend = True, is_subplots = False, reset_index = True, figsize=(15,5),
                           title='Events Data - precentile question features',
                           save_name='questions_event_features-percentile.png',
                           OVERRIDE=True)
            simple_df_plot(question_features, columns=['mistakes_normalized_entropy'],
                           kind = 'bar', is_legend = True, is_subplots = False, reset_index = True, figsize=(15,5),
                           title='Events Data - entropy', save_name='questions_event_features-normalized_entropy.png',
                           OVERRIDE=True)
            simple_df_plot(question_features,
                           columns=['n_common_mistakes(>0.10)', 'n_non_common_mistakes (<0.01 or 1 student)'],
                           kind = 'bar', is_legend = True, is_subplots = False, reset_index = True, figsize=(15,5),
                           title='Events Data - advanced question features',
                           save_name='questions_event_features-common vs non common count.png', OVERRIDE=True)
            simple_df_plot(question_features,
                           columns=['percent_common_mistakes(>0.10)', 'percent_non_common_mistakes (<0.01 or 1 student)'],
                           kind='bar', is_legend=True, is_subplots=False, reset_index=True, figsize=(15, 5),
                           title='Events Data - advanced question features',
                           save_name='questions_event_features-common vs non common percent.png',
                           OVERRIDE=True)


        return question_features

class studentEventAnalyzer(QuestionEventAnalyzer):

class SystematicEventAnalyzer(QuestionEventAnalyzer):




    def chi2_contingency_test(self, responses_to_question_1, responses_to_question_2):
        clean_response= lambda s: s[s.find(': ')+3:-2].replace("\\\\frac","").replace("}{","/")
        self.test_results_items = ['mi', 'chi2','mi_fixed', 'chi2_fixed', 'p', 'dof', 'n_students', 'n_joint_responses',
                                   'n_students_per_joint_responses',
                                   'max_n_students_joint_responses', 'joint_responses_dict']
        empty_response=[pd.Series([-1 for i in self.test_results_items],index=self.test_results_items),df(),df()]

        responses=df(pd.concat([responses_to_question_1, responses_to_question_2],axis=1)) #remove students who didn't answer one of the questions.
        responses.dropna(how='any', inplace=True)
        n_students=len(responses)
        if len(responses)<2:
            return empty_response #no joint students
        else:
            joint_responses=pd.Series(list(zip(responses[responses_to_question_1.name], responses[responses_to_question_2.name])), index=responses.index)
            observed_counts=joint_responses.value_counts()
            observed_f=observed_counts/len(joint_responses)
            observed_f.index=pd.MultiIndex.from_tuples(observed_f.index)
            observed_f=observed_f.unstack().fillna(0.)
            chi2, p, dof, expected_f = chi2_contingency(observed_f)
            g, pg, dofg, expected = chi2_contingency(observed_f, lambda_="log-likelihood")

            mi = 0.5 * g / observed_f.unstack().sum()

            observed_counts_filtered=observed_counts[observed_counts>1]
            joint_responses_dict=str(dict(observed_counts_filtered))
            n_joint_responses=len(observed_counts_filtered)
            if n_joint_responses>0:
                mi_fixed=mi
                chi2_fixed=chi2
            else:
                chi2_fixed = 0
                mi_fixed = 0
            n_students_per_joint_responses=observed_counts_filtered.sum()
            max_n_students_joint_responses=observed_counts_filtered.max()
            expected_f=df(expected_f, index=observed_f.index, columns=observed_f.columns)
            #save results to df:
            test_results=pd.Series([mi, chi2 ,mi_fixed, chi2_fixed, p ,dof ,n_students, n_joint_responses,n_students_per_joint_responses,max_n_students_joint_responses,joint_responses_dict],
                                   index=self.test_results_items)
            #return chi2, p, dof, observed_f, expected_f, n_students, mi , responses_dict
            return test_results, observed_f, expected_f

        '''expected_f=df(np.outer(marginal_frequency1,marginal_frequency2),index=marginal_frequency1.index, columns=marginal_frequency2.index).unstack()
        #expected_f.index=list(zip(expected_f.index))
        observed_vs_expected_f=pd.concat([expected_f, observed_f], axis=1)
        observed_vs_expected_f.columns = ['expected_f', 'observed_f']
        print(observed_vs_expected_f)
        temp_save=lambda  var_name, var : var.to_csv(os.path.join(OUTPUT_DIR,'temp','%s.csv' %var_name))
        for var_name, var in locals().iteritems():
            temp_save(var_name, var)

        marginal_frequency1.loc[marginal_frequency1<0.02]='other'
        raw_responses2 = self.processed_event_data.loc[self.processed_event_data.question_id == question_id2][
            ['student_id', 'clean_response', 'score']]
        raw_responses2.set_index('student_id', drop=True, inplace=True)
        correct_answers2 = set(raw_responses2.loc[raw_responses2.score == 1]['clean_response'])
        raw_responses2.loc[raw_responses2.clean_response in correct_answers2]='correct'
        raw_responses2 = raw_responses2['clean_response']
        raw_responses2[raw_responses2 in correct_answers2] = 'correct'
        raw_responses2.name = question_id2



        marginal_frequency2 = raw_responses2.value_counts() / len(raw_responses1)

        non_common_responses1=[i for i in marginal_frequency1.loc[marginal_frequency1>0.01].index if i not in correct_answers1]
        non_common_responses2 = [i for i in marginal_frequency2.loc[marginal_frequency2 > 0.01].index if i not in correct_answers2]
        print(marginal_frequency1)
        marginal_frequency2=raw_responses2.value_counts() / len(raw_responses2)


        raw_responses=pd.concat([raw_responses1,raw_responses2], axis=1)
        raw_responses.columns=[question_id1, question_id2]
        chisquare()'''

    def pairwise_question_chi2(self, questions_list, how='all_vs_all',is_plot=False,is_save_to_csv=False,print_details=False):
        question_id1 = questions_list[0]
        responses1, marginal_frequency1 = self.get_response_vector(question_id1,clean_response=True)
        results_items=['all']#['chi2', 'p', 'dof', 'n_students','mi']
        results_table=df(index=questions_list, columns=results_items)
        #\todo - continue here - make running window over question
        for q_ind in range(len(questions_list)-1):
            question_id2=questions_list[q_ind+1]
            responses2, marginal_frequency = self.get_response_vector(question_id2 , clean_response=True)
            test_results, observed_f, expected_f =self.chi2_contingency_test(responses1, responses2)


            if print_details:
                print(df(list(set(responses1.index).intersection(set(responses2.index))), columns=['---joint students:---']))
                print('----responses 1:----')
                print(responses1)
                print('----responses 2:----')
                print(responses2)
                print('----observed:----')
                print(observed_f)
                print('----expected:----')
                print(expected_f)
                #print('chi2=%s , p=%s, dof=%s, n_students=%s' %(chi2, p, dof, n_students))
            if results_items==['all']:
                results_table.columns=list(test_results.index)
                results_items=list(test_results.index)
            results_table[results_items].loc[question_id2]= test_results[results_items]
            responses1=responses2.index


                #results_table.loc[question_id2][['chi2', 'p', 'dof', 'n_students']] = df([chi2, p, dof, n_students])


        #add meta data to results table:
        print(self.meta_data.columns)
        md_index=self.meta_data[['LO_subject_index', 'LO_subsubject_index', 'LO_general_index','LO_combined_index','num_of_questions_in_lo_session','question_index']]
        md_index.index=self.meta_data.sElementID
        questions_md=md_index.loc[questions_list].drop_duplicates()
        results_table[questions_md.columns]=questions_md

        self.chi2_restuls_table=results_table



        if is_plot:
            simple_df_plot(results_table,is_subplots=True, reset_index=True, is_legend=True, save_name='chi2_restuls_all_questions.png',OVERRIDE=True)

        if is_save_to_csv:
            results_table.to_csv(os.path.join(OUTPUT_DIR,'chi2_running_pairwise_results.csv'))


        return results_table

    def all_vs_all_chi2(self,questions_list, is_plot=False, is_save_to_csv=False):

        results_items=['all']#['chi2', 'p', 'dof', 'n_students','mi']
        print('------------len questions list='+ str(len(questions_list)+'-------'))
        res = {}
        responses={}
        calculated_questions=[]
        for q1 in questions_list:
            if q1 in responses.keys():
                responses1=responses[q1]
            else:
                responses1, marginal_frequency1 = self.get_response_vector(q1,clean_response=True)
                responses[q1]=responses1
            calculated_questions.append(q1)
            print('q1:%s' %q1)
            print('q1:%s vs. %s' %(len(calculated_questions),len(questions_list)-len(calculated_questions)))
            for q2 in questions_list:
                if q2 not in calculated_questions:
                    if q2 in responses.keys():
                        responses2 = responses[q2]
                    else:
                        responses2, marginal_frequency2 = self.get_response_vector(q2,clean_response=True)
                        responses[q2] = responses2
                    test_results, observed_f, expected_f= self.chi2_contingency_test(responses1, responses2)
                    if results_items==['all']:
                        results_items=list(test_results.index)
                        for i in results_items:
                            res[i] = df(index=questions_list, columns=questions_list)
                    for i in results_items:
                        res[i][q1].loc[q2] = test_results[i]

        if is_save_to_csv:
            full_results=df(index=res[results_items[0]].unstack().index,columns=results_items)

            meta_data_columns=['LO_subject_index', 'LO_subsubject_index', 'LO_general_index', 'LO_combined_index',
                 'num_of_questions_in_lo_session', 'question_index']
            md = self.meta_data[meta_data_columns
                ].copy()
            md.index = self.meta_data.sElementID
            md.drop_duplicates(inplace=True)
            questions_md = md.loc[questions_list].drop_duplicates()
            for i in results_items:
                res[i].to_csv(os.path.join(OUTPUT_DIR,'all_vs_all_chi2_results_%s.csv' %i))
                full_results[i]=res[i].unstack()
            md_q1 = md.loc[full_results.index._get_level_values(0)]
            md_q1.columns=['%s_1' %c for c in meta_data_columns]
            md_q1.index=full_results.index
            md_q2= md.loc[full_results.index._get_level_values(1)]
            md_q2.columns = ['%s_2' % c for c in meta_data_columns]
            md_q2.index = full_results.index

            full_results=pd.concat([full_results,md_q1,md_q2],axis=1)
            full_results['same_LO']=full_results.LO_general_index_1 == full_results.LO_general_index_2
            full_results['same_subject'] = full_results.LO_subject_index_1 == full_results.LO_subject_index_2

            full_results.to_csv(os.path.join(OUTPUT_DIR,'all_vs_all_chi2_results_full.csv'))
        return res, full_results

    def select_questions(self):
        responses2, marginal_frequency2 = self.get_response_vector(question_id2)
        self.preprocess_event_data(filter_only_first_attempt=True)
        question_indeces=self.processed_event_data
        return


"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""
def main():
    QEA = QuestionEventAnalyzer()
    '''q_id1 = 'question_0d1f3326-f81e-49e5-b796-7e47151fe3f8'
    q_id2 = 'question_90b19eaf-a1b3-4f82-9e7e-fa67a570190f'
    QEA.preprocess_event_data(filter_only_first_attempt=True, filter_by_min_n_answers=2,
                              filter_by_min_n_students=2)
    questions_list = QEA.processed_questions_list
    QEA.get_common_mistakes_df(questions_list,LOs='all')'''


    #QEA.get_question_features(q_id1,is_plot_mistake_description=True, save_name=q_id+'_hist.png')
    load_results=False
    plot_grid_heat_map=True
    '''EA=EventsAnalyzer()
    EA.load_data()
    EA.get_all_questions_event_general_description(min_num_students=150, min_answers=3, max_answers=50, is_plot=True)
    print(EA.top_questions)'''

    #arrange questions by LOs:
    #lo_index = df.from_csv(os.path.join(DATA_ROOT, 'LOs_order_index_fraction.csv'))
    #\todo - CONTINUE HERE -  (1) arrange questions according to LO and question index (2) per LO- run all questions against each other (hitmap) and pairwise.
    SEA = SystematicEventAnalyzer()


    SEA.preprocess_event_data(filter_only_first_attempt=True, filter_by_min_n_answers=3,
                              filter_by_min_n_students=10)
    questions_list = SEA.processed_questions_list
    LOs='1'
            #pd.MultiIndex.from_arrays([res_index.LO_subject_index,res_index.question_index])'''
    #questions_list=questions_list[:20]
    results_items = ['all']#'['mi', 'chi2', 'p', 'dof', 'n_students']
    #results_items = ['chi2_fixed','mi_fixed', 'n_joint_responses',	'n_students_per_joint_responses',	'max_n_students_joint_responses']
    if load_results:
        full_results=df.from_csv(os.path.join(OUTPUT_DIR, 'all_vs_all_chi2_results_full.csv'), index_col=['q1','q2'])
        res={}
        for item in results_items:
            item_results=full_results[item].unstack()
            item_results.index=pd.MultiIndex.from_arrays([full_results])
            res[item]=full_results[item]

            res[item] = df.from_csv(os.path.join(OUTPUT_DIR, 'all_vs_all_chi2_results_%s.csv' % item))
    else:

        res, full_results =SEA.all_vs_all_chi2(questions_list, is_save_to_csv=True)

    for item in results_items:
        results=res[item]
        res_index = SEA.question_indexes.loc[results.index]
        results.index = res_index.question_index
        axis1 = res_index.LO_subsubject_index.reset_index(drop=True)
        axis2 = res_index.LO_subject_index.reset_index(drop=True)

        if plot_grid_heat_map:
            grid_heatmap(results, index1=axis1, index2=axis2, title=item, save_name='all_vs_all_chi2_results_%s.png' % item)



    #questions_list = [q_id1, q_id2]
    pairwise_res = SEA.all_vs_all_chi2(questions_list, is_save_to_csv=True)
    pairwise_res=SEA.pairwise_question_chi2(questions_list,is_save_to_csv=True)
    QEA=QuestionEventAnalyzer()
    QEA.agg_by_question_attempts()

    #QEA.get_all_questions_features(is_plot_by_columns=True, is_save_csv=True)

    question_ind='question_39e75740-5f1f-4347-8870-44623b9b8e07'
#'question_629275a1-9403-48f9-853d-e5c981eecce8'#''question_4c3f85a1-0470-4fbb-9183-4ea4b4e53b57'#'question_25504e71-8563-4600-aa5e-be589b5e44c5'
    QEA.get_question_event_description(question_ind, is_plot=False)
    QEA.get_question_features(question_ind,is_plot_mistake_description=True)


main()