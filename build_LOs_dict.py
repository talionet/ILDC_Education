from pandas import DataFrame as df

from settings import *

LOs_order=df.from_csv(os.path.join(DATA_ROOT,'LOs_order.csv'))
LOs_order.reset_index(inplace=True)
lo_details=LOs_order.loc[LOs_order['sTreeTitle']==TREE_TITLE].copy()


lo_details.reset_index(inplace=True,drop=True)
lo_details['general_index']=lo_details.index
lo_details['subsubject_index']=lo_details.sLOTitle.apply(lambda s: s[s.rfind(' ')+1 :])
lo_details['subsubject_index'].iloc[0]='1'

lo_details['subject']=[lo_details.sLOTitle[i].replace(lo_details.subsubject_index[i],'') for i in lo_details.index]
print(lo_details['subject'])

#lo_details.to_csv(os.path.join(DATA_ROOT,'LOs_order_with_index.csv'))

"""subjects_dict={q:i for i,q in enumerate(lo_details.subject.value_counts(sort=False).index)}
lo_details['subject_index']=[subjects_dict[q] for q in lo_details.subject]
LOs_order['subject_index']=0"""






