# -*- coding: utf-8 -*-

#Import libraries
import urllib as url
import datetime
import pytz
import pyodbc
import pandas as pd
import numpy as np
import recordlinkage
import datetime as dt
import uuid
import csv
import random, string
pd.options.mode.chained_assignment = None

existing_ids_df = pd.read_csv(r'C:\existing_data.csv')

matched_df = existing_ids_df
matched_df1 = matched_df[['CustomerId','Name','Email','Address','unique_id']]

by_max_customerId = matched_df1.groupby(['Name','Email','Address','unique_id']).CustomerId.transform(max)
matched_df1 = matched_df1.loc[matched_df1.CustomerId == by_max_customerId]

replace_values = {np.nan : 'blankprogrammed', '' : 'blankprogrammed',' ':'blankprogrammed'} 
def replace_frame_values(in_df: pd.DataFrame(), replace_values: dict) -> pd.DataFrame():
    """
    :param in_df:
    :param replace_values:
    :return:
    """
    return in_df.replace(
        {
            "Email": replace_values,
            "Name": replace_values,
            "Address": replace_values
        }
    )

matched_df1 = replace_frame_values(matched_df1, replace_values)

new_incremental_data = pd.read_csv(r'C:\new_data.csv')

new_incremental_data = replace_frame_values(new_incremental_data, replace_values)
new_incremental_data['index1'] = new_incremental_data.index

new_incremental_data = new_incremental_data.applymap(lambda s: s.lower().strip() if isinstance(s, str) else s)

new_incremental_data['isDuplicateName'] = new_incremental_data['Name'].apply(lambda x: 1 if "duplicate" in x else 0)

duplicate_clients = new_incremental_data.loc[new_incremental_data['isDuplicateName'] == 1]
duplicate_clients = duplicate_clients[['CustomerId','Name','isDuplicateName']]

duplicate_stop_words = ['dont use me','#','0','1','2','3','4','5','6','7','8','9',',','dont use',\
'use instead','!',':','this one','duplicate', '*', '(', ')',"'", 'account','please use other', 'do not use', 'acct.','acct', \
'-','acc','&','/','+','&','[',']','please','use','instead']

NameNew = []
for index, row in duplicate_clients.iterrows():
    if row['isDuplicateName'] == 1:
        match_word = row['Name']
        for item in duplicate_stop_words:
            if item in match_word:                
                match_word = match_word.replace(item,'')
        NameNew.append(match_word.strip())
    else:
        NameNew.append(row['Name'])

duplicate_clients['Name_duplicate'] = NameNew

#duplicate_clients
duplicate_clients = duplicate_clients[['CustomerId','Name_duplicate']]
new_incremental_data = pd.merge(new_incremental_data, duplicate_clients,how='left',left_on='CustomerId',right_on='CustomerId')
new_incremental_data['Name'] = np.where(new_incremental_data['isDuplicateName'] == 1, new_incremental_data['Name_duplicate'], new_incremental_data['Name'])
new_incremental_data = replace_frame_values(new_incremental_data, replace_values)
unnamed_conditions = [
    (new_incremental_data['Name'] == 'blankprogrammed') & (new_incremental_data['Email'] == 'blankprogrammed'),
    (new_incremental_data['Name'] == 'blankprogrammed') | (new_incremental_data['Email'] == 'blankprogrammed')
    ]

unnamed_values = [1,0]
new_incremental_data['isUnamedName'] = np.select(unnamed_conditions, unnamed_values)


#identifying the test clients;
new_incremental_data.shape
new_incremental_data[new_incremental_data['Name'].str.contains("test") & new_incremental_data['Name'].str.contains("\*")] 

conditions = [
    (new_incremental_data['Name'].str.contains("test") & new_incremental_data['Name'].str.contains("\*")),
    (new_incremental_data['isUnamedName'] == 1 ),
    (new_incremental_data['Name'].str.contains("test") & ~new_incremental_data['Name'].str.contains("\*") & \
        new_incremental_data['isUnamedName'] == 0),
    (~new_incremental_data['Name'].str.contains("test") & new_incremental_data['isUnamedName'] == 0 )
]

values = ['ID-00000000-0000-0000-0000-000000000000','ID-00000000-0000-0000-0000-000000000001','notTest','notTest']


new_incremental_data['testId'] = np.select(conditions, values)

print(new_incremental_data.shape)

new_incremental_data = new_incremental_data.drop_duplicates()
new_df = new_incremental_data[new_incremental_data['testId'] == 'notTest'][['CustomerId','Name','Email','Address']]
new_df = new_df.drop_duplicates()
new_df['isNewData'] = 1
matched_df1['isNewData'] = 0
matched_df2 = matched_df1[['CustomerId','Name','Email','Address','isNewData']]
final_df = pd.concat([new_df,matched_df2])
final_df = final_df.drop_duplicates()
final_df['index1'] = final_df.index
final_df.index = np.arange(len(final_df))
all_columns_df = final_df
final_df = all_columns_df[['CustomerId','Name','Email','Address']]


#passing the data through record linkage package
indexer = recordlinkage.Index()
indexer.block(left_on=['CustomerId'])
indexer.block(left_on=['Name','Email'])
indexer.block(left_on=['Name','Address'])
candidate_links = indexer.index(final_df)


# This cell can take some time to compute.
compare_cl = recordlinkage.Compare()
compare_cl.exact('CustomerId', 'CustomerId', label='CustomerId')
compare_cl.exact('Email', 'Email', label='Email')
compare_cl.exact('Name', 'Name', label='Name')
compare_cl.exact('Address', 'Address', label='Address')

features = compare_cl.compute(candidate_links, final_df)
final_df['index2'] = final_df.index

matched_id_features = features.loc[(features['CustomerId'] == 1), ['CustomerId']]
matched_id_indexes  =  matched_id_features.index
matched_id_df = matched_id_indexes.to_frame(name=['x','y'],index=False)
print('matched_id_df',matched_id_df.shape)
matched_id_df = pd.merge(matched_id_df,final_df[['index2','CustomerId']],how='left',left_on='x',right_on='index2')

matched_id_df = matched_id_df.rename(columns={"CustomerId": "CustomerId_x"})

matched_id_df = pd.merge(matched_id_df,final_df[['index2','CustomerId']],how='left',left_on='y',right_on='index2')
matched_id_df = matched_id_df.rename(columns={"CustomerId": "CustomerId_y"})

matched_id_df = matched_id_df[['CustomerId_x','CustomerId_y']]
matched_id_df.head()
matched_indexes_all= features.loc[ (features['Name'] == 1) & (features['Email']==1)\
                                  & (features['Address']==1),\
     ['Name','Email','Address']]
matched_all = matched_indexes_all.index
all_df = matched_all.to_frame(name=['x', 'y'],index=False)
print(all_df.shape)

all_df = pd.merge(all_df, final_df[['Name','Email','Address','index2']],how='left',left_on='x',right_on='index2')
all_df = all_df.loc[(all_df['Name']!='blankprogrammed')]
all_df = all_df.loc[(all_df['Email']!='blankprogrammed') | (all_df['Address']!='blankprogrammed')]
all_df = all_df[['x','y']]
all_df = pd.merge(all_df,final_df[['index2','CustomerId']],how='left',left_on='x',right_on='index2')
all_df = all_df.rename(columns={"CustomerId": "CustomerId_x"})
all_df = pd.merge(all_df,final_df[['index2','CustomerId']],how='left',left_on='y',right_on='index2')
all_df = all_df.rename(columns={"CustomerId": "CustomerId_y"})
all_df = all_df[['CustomerId_x','CustomerId_y']]
all_df.shape


name_email_features = features.loc[ (features['Name'] == 1) & (features['Email']==1) \
                                   & (features['Address']==0)]
name_email = name_email_features.index
name_email_df = name_email.to_frame(name=['x', 'y'],index=False)

name_email_df = pd.merge(name_email_df, final_df[['Name','Email','Address','index2']],how='left',left_on='x',right_on='index2')
name_email_df = name_email_df.loc[(name_email_df['Name']!='blankprogrammed')]
name_email_df = name_email_df.loc[(name_email_df['Email']!='blankprogrammed')] 
name_email_df = name_email_df[['x','y']]
print(name_email_df.shape)
name_email_df = pd.merge(name_email_df,final_df[['index2','CustomerId']],how='left',left_on='x',right_on='index2')
name_email_df = name_email_df.rename(columns={"CustomerId": "CustomerId_x"})
name_email_df = pd.merge(name_email_df,final_df[['index2','CustomerId']],how='left',left_on='y',right_on='index2')
name_email_df = name_email_df.rename(columns={"CustomerId": "CustomerId_y"})
name_email_df = name_email_df[['CustomerId_x','CustomerId_y']]
print(name_email_df.shape)


name_address_features = features.loc[ ((features['Name'] == 1) & (features['Email']==0) & \
                                       (features['Address']==1))]
name_address = name_address_features.index
name_address_df = name_address.to_frame(name=['x', 'y'],index=False)

name_address_df = pd.merge(name_address_df, final_df[['Name','Email','Address','index2']],how='left',left_on='x',right_on='index2')
name_address_df = name_address_df.loc[(name_address_df['Name']!='blankprogrammed')]
name_address_df = name_address_df.loc[(name_address_df['Address']!='blankprogrammed')]
name_address_df_col = name_address_df
name_address_df = name_address_df[['x','y']]
print(name_address_df.shape)
name_address_df = pd.merge(name_address_df,final_df[['index2','CustomerId']],how='left',left_on='x',right_on='index2')
name_address_df = name_address_df.rename(columns={"CustomerId": "CustomerId_x"})
name_address_df = pd.merge(name_address_df,final_df[['index2','CustomerId']],how='left',left_on='y',right_on='index2')
name_address_df = name_address_df.rename(columns={"CustomerId": "CustomerId_y"})
name_address_df_test = name_address_df[['CustomerId_x','CustomerId_y','x','y']]
name_address_df = name_address_df[['CustomerId_x','CustomerId_y']]
print(name_address_df.shape)


all_combinations_df_1 = pd.concat([all_df,matched_id_df])
all_combinations_df_2 = pd.concat([all_combinations_df_1,name_email_df])
all_combinations_df = pd.concat([name_address_df,all_combinations_df_2])
print(all_combinations_df.shape)
all_combinations_multiIndex = pd.MultiIndex.from_frame(all_combinations_df)
print(all_combinations_multiIndex.shape)
all_combinations_multiIndex = all_combinations_multiIndex.drop_duplicates()
print(all_combinations_multiIndex.shape)
l = all_combinations_multiIndex

# get all unique elements ("nodes") of `l'
nodes = set().union(*map(set, l))
#print(nodes)

comp = {node:{node} for node in nodes}
#print(comp)

while True:
    
    merged = False
    new_l = []  # will drop edges that have already been used in a merge
    for n1, n2 in l:
        #print(n1)
        #print(n2)
        if comp[n1] is not comp[n2]:
            # the two connected components are not the same, so merge them
            #print(comp[n1],comp[n2])
            #print(type(comp[n1]))
            new_comp = comp[n1] | comp[n2]
            #print(new_comp)
            for n in new_comp:
                #print(n)
                comp[n] = new_comp
                #print(comp[n])
            merged = True
        else:
          # keep the pair for the next iteration
            new_l.append((n1, n2))
            #print(new_l)
    if not merged:
        # all done
        break
    l = new_l

# now print all distinct connected components
final_list = []
for c in set(map(frozenset, comp.values())):
    #print(list(c))
    final_list.append(list(c))
print(len(final_list))
#print(final_list)
#distinct client indexes

import itertools
matched_indexes_single_list = []
for sub_list in range(len(final_list)):
    inner_list = final_list[sub_list]
    for item in range(len(inner_list)):
        matched_indexes_single_list.append(inner_list[item])
print(len(matched_indexes_single_list))

all_indexes_df = pd.DataFrame()
all_indexes_df['CustomerIdNew'] = matched_indexes_single_list
all_indexes_df = pd.merge(final_df['CustomerId'], all_indexes_df, how='left', left_on = 'CustomerId', right_on ='CustomerIdNew')
all_indexes_df['CustomerIdNew'] = all_indexes_df['CustomerIdNew'].replace(np.nan,'None')
unique_indexes = all_indexes_df['CustomerId'].loc[(all_indexes_df['CustomerIdNew'] == 'None')].to_list()
print(len(unique_indexes))


#appending matched index pairs and unique indexes into one single list
full_list = []
for each_item in range(len(final_list)):
    full_list.append(final_list[each_item])
for each_item in range(len(unique_indexes)):
    full_list.append(unique_indexes[each_item])
print(len(full_list))
#print(full_list)
matched_ids_dict = matched_df1[['CustomerId','unique_id']]
matched_customer_keys_df = pd.DataFrame()
matched_customer_keys_df['matched_keys'] = full_list
unique_dict: dict
matched_ids = {}
unmatched_ids = []

def convert_list(in_str: str):
    #print('func')
    in_list = in_str.replace("'", "").replace('"', '').replace('[', '').replace(']', '').replace(',', '').split(" ")
    #print(in_list)
    # out_list = [int(item) if isinstance(item, int) else item for item in in_list]
    out_list = [int(item) for item in in_list]
    #print(out_list)
    #print('func end')
    out_list.sort()
    return out_list


#checking if the unique id already exists, if yes assign the same unique id to the new customer Id
def check_element(in_list: list):
    global matched_ids
    is_match = False
    unique_key = -1
    unique_id_val = ""
    #print(in_list)
    for e in in_list:
        #print(e)
        if e in unique_dict.keys():
            #print('match found')
            is_match = True
            unique_key = e
            #print(unique_key)
            unique_id_val = unique_dict[unique_key]
            #print(unique_id_val)
    if is_match:
        if unique_id_val not in matched_ids.keys():
            matched_ids[unique_id_val] = {"matching_key": unique_key, "dupe_keys": in_list}
            #print(matched_ids[unique_id_val])
        else:
            new_id = matched_ids[unique_id_val]["dupe_keys"] + in_list
            matched_ids[unique_id_val] = {"matching_key": unique_key, "dupe_keys": new_id}
            #print(matched_ids[unique_id_val])
        #print('macthed_list')
        #print(matched_ids)
    else:
        #print('unmatched_list')
        unmatched_ids.append(in_list)
        #print(unmatched_ids)


global unique_dict
print('start time: ',dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S'))
dupes_list = [convert_list(str(col)) for col in matched_customer_keys_df["matched_keys"]]
#print(dupes_list)
unique_dict = {
    int(col[0]): col[1] for col in matched_ids_dict[["CustomerId", "unique_id"]].values #dict:key value pair
    }
_ = [check_element(element) for element in dupes_list]
print(f"Matched Ids Count:  {len(matched_ids):,}")
print('end time: ',dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S'))
matched_ids_df = pd.DataFrame.from_dict(matched_ids)
matched_ids_trans_df = matched_ids_df.T
matched_ids_trans_df.reset_index(inplace=True)
matched_ids_trans_df = matched_ids_trans_df.rename(columns = {'index':'uid'})
matched_ids_trans_df = matched_ids_trans_df[['uid','dupe_keys']]


#the new customers found in the existing unique ids
unique_matched_ids_df = matched_ids_trans_df.explode('dupe_keys').drop_duplicates().rename(columns={'dupe_keys':'customerKey'})
existing_uids = matched_df['unique_id'].drop_duplicates().to_list()
new_existing_uids = [v.replace('ID-','') for v in existing_uids] 


#checking id the unique id already exists
def if_uid_exists(new_counter,loop_counter): 
    new_ids_list = []
    new_list_var = []
    new_list = []
    match_df = pd.DataFrame()
    new_list = [str(uuid.uuid4()) for i in range(new_counter)] 
    print(len(new_list))
    new_df = pd.DataFrame()
    new_df['new_uid'] = new_list
    if loop_counter ==0:
        old_uid_df = pd.DataFrame()
        old_uid_df['old_uid'] = new_existing_uids
        match_df = pd.merge(old_uid_df, new_df, how='outer', left_on='old_uid', right_on='new_uid')
    else:
        match_df = pd.merge(existing_uid_df, new_df, how='outer', left_on='old_uid', right_on='new_uid')
    match_df = match_df.replace(np.nan,'None')
    print(match_df.shape)
    cnt = match_df.loc[(match_df['old_uid'] != 'None') & (match_df['new_uid'] != 'None')].shape[0]
    new_list_var = match_df['new_uid'].loc[(match_df['old_uid'] == 'None')].values.tolist()
    #print(cnt)
    if cnt> 0:
        loop_counter += 1
        print('cnt greater')
        newly_added_ids = match_df['new_uid'].loc[(match_df['old_uid'] == 'None') & (match_df['new_uid'] != 'None') ]
        _ = [new_ids_list.append(z) for z in new_list_var]
        old_uid_df = pd.concat([old_uid_df,newly_added_ids])
        return if_uid_exists(cnt,loop_counter)
    else: 
        print('done')
        #new_ids_list = match_df['new_uid'].loc[(match_df['old_uid'] == 'None')].values.tolist()       
        _ = [new_ids_list.append(z) for z in new_list_var]
        print(len(new_ids_list))
        #print(new_ids_list[0:100])
        return new_ids_list
    #old_uid_df = existing_uid_df
    
print('start time: ',dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S'))
new = []
brand_new_ids = []
brand_new_ids1 = []
new_counter = len(unmatched_ids)
loop_counter = 0
new = if_uid_exists(new_counter,loop_counter)
_ = [brand_new_ids.append(i) for i in new]
len(brand_new_ids)
print('start time: ',dt.datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S'))

uid_final_list = ['ID-' + x for x in brand_new_ids]
matched_ids_trans_df = matched_ids_trans_df.rename(columns={'uid':'unique_id', 'dupe_keys':'matched_keys'})
matched_pairs_df = pd.DataFrame()
matched_pairs_df['unique_id'] = uid_final_list
matched_pairs_df['matched_keys'] = unmatched_ids
matched_pairs_df
final_incremental_df = pd.concat([matched_ids_trans_df,matched_pairs_df])
unique_ids_df = final_incremental_df.explode('matched_keys')
unique_ids_df = unique_ids_df.drop_duplicates()
new_incremental_data['CustomerId'] = new_incremental_data['CustomerId'].astype(int)
result = pd.merge(new_incremental_data, unique_ids_df, how="left", left_on="CustomerId",right_on = 'matched_keys')
print(result.shape)


result['unique_id'] = np.where(result['testId'] != 'notTest',result['testId'],result['unique_id'])
#print(result)
replace_values = {'blankprogrammed':np.nan} 
result = result.replace({"Email":replace_values, "Name": replace_values,\
                  "Address":replace_values}) 
result = result[['CustomerId','Email','Address','Name','isDuplicateName','Name_duplicate','isUnamed','testId','unique_id']]
result.ro_csv(r'C:\results.csv')

