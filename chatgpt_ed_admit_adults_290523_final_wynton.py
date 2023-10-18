import os
import openai
import tiktoken
import pandas as pd
import re

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


df = pd.read_parquet('path/chatgpt_admit_ed_ed_notes_edprovider_adults_filtered_master_290523.parquet')
df['encounterkey'] = df['encounterkey'].astype(str)
df['note_text'] = df['note_text'].astype(str)

#Make new admitted column
df.loc[df['hospitaladmissionkey'] == '-1', 'admitted'] = 0
df.loc[df['hospitaladmissionkey'] != '-1', 'admitted'] = 1

print(df.shape)
print(df.encounterkey.nunique())
print(df.admitted.value_counts())

##Conduct minimal preprocessing
#Remove '\n' (as upon inspection these appear to be randomly inserted into text)
df["note_text_processed"] = [re.sub(r'\n', '', s) for s in df["note_text"]]
#Confirm this
print(df[df['note_text_processed'].str.contains('\n')].shape, '- should be 0')
#Keep the deidentification ***** for now

#Remove extra spaces
def remove_extra_spaces(text):
    # Use regular expressions to replace multiple spaces with a single space
    return re.sub(' +', ' ', text)
df['note_text_processed'] = df['note_text_processed'].apply(remove_extra_spaces)

# ### Remove duplicate encounters (i.e where duplicate notes exist)
#Check for duplicates
print(len(df[df['encounterkey'].duplicated(keep = False)]))

#Examine duplicates
pd.set_option('display.max_colwidth', None)
df[df['encounterkey'].duplicated(keep = False)][['deid_service_date','encounterkey','note_text']].head(10)
#Many are smaller length attestation notes (with identical deid_service_date); others are follow up
#notes (e.g following completion of scans) - hence drop_duplicates keeping first note on deid_service_date, than on word count
#(keeping the longest note if deid_service_date is the same)

#Sort by note time and then length per above
df['deid_service_date'] = pd.to_datetime(df['deid_service_date'])
df['note_length_words'] = df['note_text_processed'].str.len()
df = df.sort_values(['encounterkey', 'deid_service_date', 'note_length_words'], ascending=[True, True, False])

#Confirm this
pd.set_option('display.max_colwidth', 40)
df[df.encounterkey.duplicated(keep = False)].head(14)
#Confirmed

print(df.shape)
df = df.drop_duplicates(subset = 'encounterkey', keep = 'first')
print(df.shape)

# ### Segment note_text_processed
import re
def extract_text(text, start_pattern, end_pattern):
    start_regex = re.compile('|'.join(start_pattern))
    end_regex = re.compile('|'.join(end_pattern))

    try:
        start_match = start_regex.search(text) 
        end_match = end_regex.search(text) 
        start = start_match.start()
        end = end_match.start()
        result = text[start:end]
    except AttributeError:
        result = 'unable_to_extract'
    return result

def extract_initialassessment_to_end(text, initialassessment, edcourse):
    #Search first for 'Initial Assessment' and if present select text from there to end
    #Otherwise search for 'ED Course' and do the same
    #Otherwise return 'unable_to_extract'
    initialassessment_regex = re.compile('|'.join(initialassessment))
    edcourse_regex = re.compile('|'.join(edcourse))
    
    start_match = initialassessment_regex.search(text)
    if start_match is None:
        start_match = edcourse_regex.search(text)
    if start_match is None:
        return 'unable_to_extract'
    start = start_match.start()
    return text[start:]

# Apply the function and create new columns for each section

#Create list of upper/lower case variations of desired note heading
#Note that e.g all lowercase 'initial assessment' has several false positives, so settle for only 1) First letter caps and 2) all caps
chiefcomplaint = ['Chief Complaint', 'CHIEF COMPLAINT']
physicalexam = ['Physical Exam', 'PHYSICAL EXAM']
initialassessment = ['Initial Assessment', 'INITIAL ASSESSMENT']
edcourse = ['ED Course', 'ED course', 'ED COURSE']

df['history_text'] = df['note_text_processed'].apply(lambda x: extract_text(x, chiefcomplaint, physicalexam)) 

df['examination_text'] = df['note_text_processed'].apply(lambda x: extract_text(x, physicalexam, initialassessment) 
                                               if any(s in x for s in initialassessment) else extract_text(x, physicalexam, edcourse) 
                                               if any(s in x for s in edcourse) else 'unable_to_extract')

df['assessment_plan_text'] = df['note_text_processed'].apply(lambda x: extract_initialassessment_to_end(x, initialassessment, edcourse))


print(df.note_text_processed.isnull().sum())
print(df.history_text.isnull().sum())
print(df.examination_text.isnull().sum())
print(df.assessment_plan_text.isnull().sum())

print((df['history_text'] == 'unable_to_extract').sum())
print((df['examination_text'] == 'unable_to_extract').sum())
print((df['assessment_plan_text'] == 'unable_to_extract').sum())


#Count any values of '' (e.g if the second regex pattern comes before the first)
print(df[df['history_text'] == ''].shape)
print(df[df['examination_text'] == ''].shape)
print(df[df['assessment_plan_text'] == ''].shape)

#Count number of null values (shouldn't be any - they should all be 'unable_to_extract')
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print(df[text].isnull().sum())
#0 for each

# ### Get token count for each text column
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

df['note_text_processed_tokens'] = df['note_text_processed'].apply(lambda x: num_tokens_from_string(x))
print('Next')
df['history_text_tokens'] = df['history_text'].apply(lambda x: num_tokens_from_string(x))
print('Next')
df['examination_text_tokens'] = df['examination_text'].apply(lambda x: num_tokens_from_string(x))
print('Next')
df['assessment_plan_text_tokens'] = df['assessment_plan_text'].apply(lambda x: num_tokens_from_string(x))

print(df['note_text_processed_tokens'].sum())
print('ChatGPT cost (full note) @ 0.002 per 1k tokens:', 0.002/1000*df['note_text_processed_tokens'].sum())

print(df['history_text_tokens'].sum())
print('ChatGPT cost (history only) @ 0.002 per 1k tokens:', 0.002/1000*df['history_text_tokens'].sum())

print(df['examination_text_tokens'].sum())
print('ChatGPT cost (examination only) @ 0.002 per 1k tokens:', 0.002/1000*df['examination_text_tokens'].sum())

print(df['assessment_plan_text_tokens'].sum())
print('ChatGPT cost (assessment/plan only) @ 0.002 per 1k tokens:', 0.002/1000*df['assessment_plan_text_tokens'].sum())

print(df.shape)
print(df.columns)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
#Check nulls - these are all the '' specified above where the second regex pattern comes prior to rather than after the first.
print(df.isnull().sum())

for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print(df[text].isnull().sum())

# ### Create df_refined
##Create a modified version of the master df which contains nulls etc excluded:
df_refined = df.copy()

#Remove null values ['history_text', 'examination_text', 'assessment_plan_text']
print(df_refined.shape)
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing nulls from', text)
    df_refined = df_refined[df_refined[text].notnull()]
    print(df_refined.shape)

#Similarly, remove 'unable_to_extract' from ['history_text', 'examination_text', 'assessment_plan_text']
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing unable_to_extract from', text)
    df_refined = df_refined[df_refined[text] != 'unable_to_extract']
    print(df_refined.shape)
    
#For completeness, remove '' from ['history_text', 'examination_text', 'assessment_plan_text']
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing unable_to_extract from', text)
    df_refined = df_refined[df_refined[text] != '']
    print(df_refined.shape)

##Examine number of notes with >4000 tokens (to exclude because of ChatGPT context window)
print(df_refined[df_refined['note_text_processed_tokens'] > 4000].shape)
print(df_refined[df_refined['note_text_processed_tokens'] < 4000].shape)
#Hence if excluding notes >4000 tokens in length (full note), will only exclude 2539 of 264889

df_refined = df_refined[df_refined['note_text_processed_tokens'] < 4000]

##Count number of tokens for history and examination sections combined:
df_refined['history_examination_text_tokens'] = df_refined['history_text_tokens'] + df_refined['examination_text_tokens']


#Set binary labels for imaging orders
df_refined.loc[df_refined['imaging_orders'].isnull(), 'imaging_orders_binary'] = 'N'
df_refined.loc[df_refined['imaging_orders'].notnull(), 'imaging_orders_binary'] = 'Y'

print(df_refined['imaging_orders_binary'].value_counts())

# ## Create admit, balanced sample dataset
initial_prompt_history_examination_admit1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'. Please do not return any additional explanation."
initial_prompt_history_examination_admit2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Only suggest admission to hospital if absolutely required. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'. Please do not return any additional explanation."
initial_prompt_history_examination_admit3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Only suggest admission to hospital if absolutely required. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'."
initial_prompt_history_examination_admit4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Let's think step by step. Only suggest admission to hospital if absolutely required. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'."

print(len(initial_prompt_history_examination_admit1))
print(len(initial_prompt_history_examination_admit2))
print(len(initial_prompt_history_examination_admit3))
print(len(initial_prompt_history_examination_admit4))

df_admit_10000 = pd.concat([df_refined[df_refined['admitted'] == 1.0].sample(5000, random_state = 7),
                              df_refined[df_refined['admitted'] == 0.0].sample(5000, random_state = 7)],
                             axis = 0)
print(df_admit_10000.admitted.value_counts())

df_admit_10000['prompt_history_examination1'] = initial_prompt_history_examination_admit1 + '  \n\n"""' + df_admit_10000['history_text'] + ' ' + df_admit_10000['examination_text'] + '"""'
df_admit_10000['prompt_history_examination2'] = initial_prompt_history_examination_admit2 + '  \n\n"""' + df_admit_10000['history_text'] + ' ' + df_admit_10000['examination_text'] + '"""'
df_admit_10000['prompt_history_examination3'] = initial_prompt_history_examination_admit3 + '  \n\n"""' + df_admit_10000['history_text'] + ' ' + df_admit_10000['examination_text'] + '"""'
df_admit_10000['prompt_history_examination4'] = initial_prompt_history_examination_admit4 + '  \n\n"""' + df_admit_10000['history_text'] + ' ' + df_admit_10000['examination_text'] + '"""'

import csv
df_admit_10000.to_csv('chatgpt_ed_admit_adults_master_processed_df_admit_10000.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')


# ## Create radiology, balanced sample dataset
initial_prompt_history_examination_radiology1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'. Please do not return any additional explanation."
initial_prompt_history_examination_radiology2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Only suggest radiological investigation if absolutely required. Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'. Please do not return any additional explanation."
initial_prompt_history_examination_radiology3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Only suggest radiological investigation if absolutely required. Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'."
initial_prompt_history_examination_radiology4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Let's think step by step. Only suggest radiological investigation if absolutely required. Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'."

print(len(initial_prompt_history_examination_radiology1))
print(len(initial_prompt_history_examination_radiology2))
print(len(initial_prompt_history_examination_radiology3))
print(len(initial_prompt_history_examination_radiology4))

df_radiology_10000 = pd.concat([df_refined[df_refined['imaging_orders_binary'] == 'Y'].sample(5000, random_state = 7),
                              df_refined[df_refined['imaging_orders_binary'] == 'N'].sample(5000, random_state = 7)],
                             axis = 0)
print(df_radiology_10000.imaging_orders_binary.value_counts())

df_radiology_10000['prompt_history_examination1'] = initial_prompt_history_examination_radiology1 + '  \n\n"""' + df_radiology_10000['history_text'] + ' ' + df_radiology_10000['examination_text'] + '"""'
df_radiology_10000['prompt_history_examination2'] = initial_prompt_history_examination_radiology2 + '  \n\n"""' + df_radiology_10000['history_text'] + ' ' + df_radiology_10000['examination_text'] + '"""'
df_radiology_10000['prompt_history_examination3'] = initial_prompt_history_examination_radiology3 + '  \n\n"""' + df_radiology_10000['history_text'] + ' ' + df_radiology_10000['examination_text'] + '"""'
df_radiology_10000['prompt_history_examination4'] = initial_prompt_history_examination_radiology4 + '  \n\n"""' + df_radiology_10000['history_text'] + ' ' + df_radiology_10000['examination_text'] + '"""'

import csv
df_radiology_10000.to_csv('chatgpt_ed_admit_adults_master_processed_df_radiology_10000.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')


# ## Create antibiotics, balanced sample dataset
initial_prompt_history_examination_antibiotics1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'. Please do not return any additional explanation."
initial_prompt_history_examination_antibiotics2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Only suggest antibiotics if absolutely required. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'. Please do not return any additional explanation."
initial_prompt_history_examination_antibiotics3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Only suggest antibiotics if absolutely required. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'."
initial_prompt_history_examination_antibiotics4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Let's think step by step. Only suggest antibiotics if absolutely required. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'."

print(len(initial_prompt_history_examination_antibiotics1))
print(len(initial_prompt_history_examination_antibiotics2))
print(len(initial_prompt_history_examination_antibiotics3))
print(len(initial_prompt_history_examination_antibiotics4))

df_antibiotics_10000 = pd.concat([df_refined[df_refined['abx_ordered_ed'] == 'Y'].sample(5000, random_state = 7),
                              df_refined[df_refined['abx_ordered_ed'] == 'N'].sample(5000, random_state = 7)],
                             axis = 0)
print(df_antibiotics_10000.abx_ordered_ed.value_counts())

df_antibiotics_10000['prompt_history_examination1'] = initial_prompt_history_examination_antibiotics1 + '  \n\n"""' + df_antibiotics_10000['history_text'] + ' ' + df_antibiotics_10000['examination_text'] + '"""'
df_antibiotics_10000['prompt_history_examination2'] = initial_prompt_history_examination_antibiotics2 + '  \n\n"""' + df_antibiotics_10000['history_text'] + ' ' + df_antibiotics_10000['examination_text'] + '"""'
df_antibiotics_10000['prompt_history_examination3'] = initial_prompt_history_examination_antibiotics3 + '  \n\n"""' + df_antibiotics_10000['history_text'] + ' ' + df_antibiotics_10000['examination_text'] + '"""'
df_antibiotics_10000['prompt_history_examination4'] = initial_prompt_history_examination_antibiotics4 + '  \n\n"""' + df_antibiotics_10000['history_text'] + ' ' + df_antibiotics_10000['examination_text'] + '"""'

import csv
df_antibiotics_10000.to_csv('chatgpt_ed_admit_adults_master_processed_df_antibiotics_10000.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')

# ## Create single n=1000, unbalanced sample dataset
##Examine current distributions:
print(df_refined.shape)

for label in ['admitted', 'imaging_orders_binary', 'abx_ordered_ed']:
    print(label)
    print(df_refined[label].value_counts(), '\n')

df_unbalanced_1000 = df_refined.sample(1000, random_state = 7)

##Examine sample distributions:
for label in ['admitted', 'imaging_orders_binary', 'abx_ordered_ed']:
    print(label)
    print(df_unbalanced_1000[label].value_counts(), '\n')

##Create separate datasets for each task (so can have the same prompt structure)
df_admit_unbalanced_1000 = df_unbalanced_1000[['patientdurablekey', 'edvisitkey', 'patientkey', 'hospitaladmissionkey',
       'encounterkey', 'admitted', 'history_text', 'examination_text', 'history_examination_text_tokens']]
df_radiology_unbalanced_1000 = df_unbalanced_1000[['patientdurablekey', 'edvisitkey', 'patientkey', 'hospitaladmissionkey',
       'encounterkey', 'imaging_orders_binary', 'history_text', 'examination_text', 'history_examination_text_tokens']]
df_antibiotics_unbalanced_1000 = df_unbalanced_1000[['patientdurablekey', 'edvisitkey', 'patientkey', 'hospitaladmissionkey',
       'encounterkey', 'abx_ordered_ed', 'history_text', 'examination_text', 'history_examination_text_tokens']]

##df_admit_unbalanced_1000
initial_prompt_history_examination_admit1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'. Please do not return any additional explanation."
initial_prompt_history_examination_admit2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Only suggest admission to hospital if absolutely required. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'. Please do not return any additional explanation."
initial_prompt_history_examination_admit3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Only suggest admission to hospital if absolutely required. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'."
initial_prompt_history_examination_admit4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Let's think step by step. Only suggest admission to hospital if absolutely required. Please return one of two answers: '0: Patient should not be admitted to hospital' '1: Patient should be admitted to hospital'."

df_admit_unbalanced_1000['prompt_history_examination1'] = initial_prompt_history_examination_admit1 + '  \n\n"""' + df_admit_unbalanced_1000['history_text'] + ' ' + df_admit_unbalanced_1000['examination_text'] + '"""'
df_admit_unbalanced_1000['prompt_history_examination2'] = initial_prompt_history_examination_admit2 + '  \n\n"""' + df_admit_unbalanced_1000['history_text'] + ' ' + df_admit_unbalanced_1000['examination_text'] + '"""'
df_admit_unbalanced_1000['prompt_history_examination3'] = initial_prompt_history_examination_admit3 + '  \n\n"""' + df_admit_unbalanced_1000['history_text'] + ' ' + df_admit_unbalanced_1000['examination_text'] + '"""'
df_admit_unbalanced_1000['prompt_history_examination4'] = initial_prompt_history_examination_admit4 + '  \n\n"""' + df_admit_unbalanced_1000['history_text'] + ' ' + df_admit_unbalanced_1000['examination_text'] + '"""'


##df_radiology_unbalanced_1000
initial_prompt_history_examination_radiology1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'. Please do not return any additional explanation."
initial_prompt_history_examination_radiology2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Only suggest radiological investigation if absolutely required. Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'. Please do not return any additional explanation."
initial_prompt_history_examination_radiology3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Only suggest radiological investigation if absolutely required. Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'."
initial_prompt_history_examination_radiology4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Let's think step by step. Only suggest radiological investigation if absolutely required. Please return one of two answers: '0: Patient does not require radiological investigation' '1: Patient requires radiological investigation'."

df_radiology_unbalanced_1000['prompt_history_examination1'] = initial_prompt_history_examination_radiology1 + '  \n\n"""' + df_radiology_unbalanced_1000['history_text'] + ' ' + df_radiology_unbalanced_1000['examination_text'] + '"""'
df_radiology_unbalanced_1000['prompt_history_examination2'] = initial_prompt_history_examination_radiology2 + '  \n\n"""' + df_radiology_unbalanced_1000['history_text'] + ' ' + df_radiology_unbalanced_1000['examination_text'] + '"""'
df_radiology_unbalanced_1000['prompt_history_examination3'] = initial_prompt_history_examination_radiology3 + '  \n\n"""' + df_radiology_unbalanced_1000['history_text'] + ' ' + df_radiology_unbalanced_1000['examination_text'] + '"""'
df_radiology_unbalanced_1000['prompt_history_examination4'] = initial_prompt_history_examination_radiology4 + '  \n\n"""' + df_radiology_unbalanced_1000['history_text'] + ' ' + df_radiology_unbalanced_1000['examination_text'] + '"""'


##df_antibiotics_unbalanced_1000
initial_prompt_history_examination_antibiotics1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'. Please do not return any additional explanation."
initial_prompt_history_examination_antibiotics2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Only suggest antibiotics if absolutely required. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'. Please do not return any additional explanation."
initial_prompt_history_examination_antibiotics3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Only suggest antibiotics if absolutely required. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'."
initial_prompt_history_examination_antibiotics4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Let's think step by step. Only suggest antibiotics if absolutely required. Please return one of two answers: '0: Patient does not require antibiotics' '1: Patient requires antibiotics'."

df_antibiotics_unbalanced_1000['prompt_history_examination1'] = initial_prompt_history_examination_antibiotics1 + '  \n\n"""' + df_antibiotics_unbalanced_1000['history_text'] + ' ' + df_antibiotics_unbalanced_1000['examination_text'] + '"""'
df_antibiotics_unbalanced_1000['prompt_history_examination2'] = initial_prompt_history_examination_antibiotics2 + '  \n\n"""' + df_antibiotics_unbalanced_1000['history_text'] + ' ' + df_antibiotics_unbalanced_1000['examination_text'] + '"""'
df_antibiotics_unbalanced_1000['prompt_history_examination3'] = initial_prompt_history_examination_antibiotics3 + '  \n\n"""' + df_antibiotics_unbalanced_1000['history_text'] + ' ' + df_antibiotics_unbalanced_1000['examination_text'] + '"""'
df_antibiotics_unbalanced_1000['prompt_history_examination4'] = initial_prompt_history_examination_antibiotics4 + '  \n\n"""' + df_antibiotics_unbalanced_1000['history_text'] + ' ' + df_antibiotics_unbalanced_1000['examination_text'] + '"""'


##Save to csv:
import csv
df_admit_unbalanced_1000.to_csv('path/chatgpt_ed_admit_adults_master_processed_df_admit_unbalanced_1000.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
df_radiology_unbalanced_1000.to_csv('path/chatgpt_ed_admit_adults_master_processed_df_radiology_unbalanced_1000.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
df_antibiotics_unbalanced_1000.to_csv('path/chatgpt_ed_admit_adults_master_processed_df_antibiotics_unbalanced_1000.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')

# ## Create n=200 balanced dataset for manual labelling
def retrieve_n_samples_balanced(df, n_samples, outcome_variable):
    half_n_samples = int(n_samples/2)
    print(half_n_samples)
    df_negative = df[df[outcome_variable] == 0]
    df_positive = df[df[outcome_variable] == 1]
    
    df_negative_sample = df_negative.sample(n = half_n_samples, replace = False, random_state = 7)
    df_positive_sample = df_positive.sample(n = half_n_samples, replace = False, random_state = 7)
    
    df_return = pd.concat([df_negative_sample, df_positive_sample])
    
    
    print(df_return.shape)
    print(df_return[outcome_variable].value_counts())
    df_return = df_return[['patientdurablekey', 'edvisitkey', 'patientkey', 'hospitaladmissionkey',
       'encounterkey', 'prompt_history_examination1']]
    
    #Shuffle df
    df_return = df_return.sample(frac=1).reset_index(drop=True)
    
    return df_return
    
df_admit_10000_fixed_200_sample = retrieve_n_samples_balanced(df_admit_10000, 200, 'admitted')
df_radiology_10000_fixed_200_sample = retrieve_n_samples_balanced(df_radiology_10000, 200, 'imaging_orders_binary')
df_antibiotics_10000_fixed_200_sample = retrieve_n_samples_balanced(df_antibiotics_10000, 200, 'abx_ordered_ed')

df_admit_10000_fixed_200_sample.to_csv('path/chatgpt_ed_admit_adults_290523_results\\df_admit_10000_fixed_200_sample.csv')
df_radiology_10000_fixed_200_sample.to_csv('path/chatgpt_ed_admit_adults_290523_results\\df_radiology_10000_fixed_200_sample.csv')
df_antibiotics_10000_fixed_200_sample.to_csv('path/chatgpt_ed_admit_adults_290523_results\\df_antibiotics_10000_fixed_200_sample.csv')


# ## Sensitivity analysis: create reversed n = 200 dataset using same sample as manually labelled dfs

##Do reversed dataset (as sensitivity analysis) to confirm that the order does not matter:
#Take the same n=200 balanced dataset that was manually labelled

##Import n=200 subsamples 
import pandas as pd
df_admit_10000_fixed_200_sample_reversed = pd.read_csv('path/chatgpt_ed_admit_adults_290523_results\\df_admit_10000_fixed_200_sample_classified.csv', index_col = 0)
df_radiology_10000_fixed_200_sample_reversed = pd.read_csv('path/chatgpt_ed_admit_adults_290523_results\\df_radiology_10000_fixed_200_sample_classified.csv', index_col = 0)
df_antibiotics_10000_fixed_200_sample_reversed = pd.read_csv('path/chatgpt_ed_admit_adults_290523_results\\df_antibiotics_10000_fixed_200_sample_classified.csv', index_col = 0)

##Overwrite the original (non-reversed) prompts with reversed prompts

##df_admit_reversed
initial_prompt_history_examination_admit_reversed1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Please return one of two answers: '1: Patient should be admitted to hospital' '0: Patient should not be admitted to hospital'. Please do not return any additional explanation."
initial_prompt_history_examination_admit_reversed2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Only suggest admission to hospital if absolutely required. Please return one of two answers: '1: Patient should be admitted to hospital' '0: Patient should not be admitted to hospital'. Please do not return any additional explanation."
initial_prompt_history_examination_admit_reversed3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Only suggest admission to hospital if absolutely required. Please return one of two answers: '1: Patient should be admitted to hospital' '0: Patient should not be admitted to hospital'."
initial_prompt_history_examination_admit_reversed4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient should be admitted to hospital. Let's think step by step. Only suggest admission to hospital if absolutely required. Please return one of two answers: '1: Patient should be admitted to hospital' '0: Patient should not be admitted to hospital'."

df_admit_10000_fixed_200_sample_reversed['prompt_history_examination1'] = initial_prompt_history_examination_admit_reversed1 + '  \n\n"""' + df_admit_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_admit_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_admit_10000_fixed_200_sample_reversed['prompt_history_examination2'] = initial_prompt_history_examination_admit_reversed2 + '  \n\n"""' + df_admit_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_admit_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_admit_10000_fixed_200_sample_reversed['prompt_history_examination3'] = initial_prompt_history_examination_admit_reversed3 + '  \n\n"""' + df_admit_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_admit_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_admit_10000_fixed_200_sample_reversed['prompt_history_examination4'] = initial_prompt_history_examination_admit_reversed4 + '  \n\n"""' + df_admit_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_admit_10000_fixed_200_sample_reversed['examination_text'] + '"""'


##df_radiology_reversed
initial_prompt_history_examination_radiology_reversed1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Please return one of two answers: '1: Patient requires radiological investigation' '0: Patient does not require radiological investigation'. Please do not return any additional explanation."
initial_prompt_history_examination_radiology_reversed2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Only suggest radiological investigation if absolutely required. Please return one of two answers: '1: Patient requires radiological investigation' '0: Patient does not require radiological investigation'. Please do not return any additional explanation."
initial_prompt_history_examination_radiology_reversed3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Only suggest radiological investigation if absolutely required. Please return one of two answers: '1: Patient requires radiological investigation' '0: Patient does not require radiological investigation'."
initial_prompt_history_examination_radiology_reversed4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires radiological investigation (e.g X-ray, ultrasound scan, CT scan or MRI scan). Let's think step by step. Only suggest radiological investigation if absolutely required. Please return one of two answers: '1: Patient requires radiological investigation' '0: Patient does not require radiological investigation'."

df_radiology_10000_fixed_200_sample_reversed['prompt_history_examination1'] = initial_prompt_history_examination_radiology_reversed1 + '  \n\n"""' + df_radiology_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_radiology_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_radiology_10000_fixed_200_sample_reversed['prompt_history_examination2'] = initial_prompt_history_examination_radiology_reversed2 + '  \n\n"""' + df_radiology_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_radiology_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_radiology_10000_fixed_200_sample_reversed['prompt_history_examination3'] = initial_prompt_history_examination_radiology_reversed3 + '  \n\n"""' + df_radiology_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_radiology_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_radiology_10000_fixed_200_sample_reversed['prompt_history_examination4'] = initial_prompt_history_examination_radiology_reversed4 + '  \n\n"""' + df_radiology_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_radiology_10000_fixed_200_sample_reversed['examination_text'] + '"""'

##df_antibiotics_reversed
initial_prompt_history_examination_antibiotics_reversed1 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Please return one of two answers: '1: Patient requires antibiotics' '0: Patient does not require antibiotics'. Please do not return any additional explanation."
initial_prompt_history_examination_antibiotics_reversed2 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Only suggest antibiotics if absolutely required. Please return one of two answers: '1: Patient requires antibiotics' '0: Patient does not require antibiotics'. Please do not return any additional explanation."
initial_prompt_history_examination_antibiotics_reversed3 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Only suggest antibiotics if absolutely required. Please return one of two answers: '1: Patient requires antibiotics' '0: Patient does not require antibiotics'."
initial_prompt_history_examination_antibiotics_reversed4 = "You are an Emergency Department physician. Below are the symptoms and clinical examination findings of a patient presenting to the Emergency Department. Please return whether the patient requires antibiotics. Let's think step by step. Only suggest antibiotics if absolutely required. Please return one of two answers: '1: Patient requires antibiotics' '0: Patient does not require antibiotics'."

df_antibiotics_10000_fixed_200_sample_reversed['prompt_history_examination1'] = initial_prompt_history_examination_antibiotics_reversed1 + '  \n\n"""' + df_antibiotics_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_antibiotics_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_antibiotics_10000_fixed_200_sample_reversed['prompt_history_examination2'] = initial_prompt_history_examination_antibiotics_reversed2 + '  \n\n"""' + df_antibiotics_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_antibiotics_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_antibiotics_10000_fixed_200_sample_reversed['prompt_history_examination3'] = initial_prompt_history_examination_antibiotics_reversed3 + '  \n\n"""' + df_antibiotics_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_antibiotics_10000_fixed_200_sample_reversed['examination_text'] + '"""'
df_antibiotics_10000_fixed_200_sample_reversed['prompt_history_examination4'] = initial_prompt_history_examination_antibiotics_reversed4 + '  \n\n"""' + df_antibiotics_10000_fixed_200_sample_reversed['history_text'] + ' ' + df_antibiotics_10000_fixed_200_sample_reversed['examination_text'] + '"""'

import csv
df_admit_10000_fixed_200_sample_reversed.to_csv('path/chatgpt_ed_admit_adults_290523_results\\reversed_balanced\\df_admit_10000_fixed_200_sample_reversed.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
df_radiology_10000_fixed_200_sample_reversed.to_csv('path/chatgpt_ed_admit_adults_290523_results\\reversed_balanced\\df_radiology_10000_fixed_200_sample_reversed.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
df_antibiotics_10000_fixed_200_sample_reversed.to_csv('path/chatgpt_ed_admit_adults_290523_results\\reversed_balanced\\df_antibiotics_10000_fixed_200_sample_reversed.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')


# ## Run through Versa API
import openai
import os
import re
import json
import base64
import datetime
import requests
import urllib.parse
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

load_dotenv('.env')
API_KEY = os.environ.get('STAGE_API_KEY')
API_VERSION = os.environ.get('API_VERSION')
RESOURCE_ENDPOINT = os.environ.get('RESOURCE_ENDPOINT')

openai.api_type = "azure"  
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT  # May change depending on which server and Mulesoft key you use
openai.api_version = '2023-03-15-preview'

deployment_name='gpt-35-turbo'

def run_chatgpt_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages = [
                {"role": "user", "content": prompt}
            ],
            n=1,
            stop=None,
            temperature=0,
            )
    except:
        response = 'Error_with_API_CYKW'
    return response

def retrieve_content_from_response_json2(x):
    try:
        return json.loads(str(x))['choices'][0]['message']['content']
    except:
        return 'Error_with_API_CYKW'

def retrieve_label(x):
    if '0' in x:
        label = '0'
    elif '1' in x:
        label = '1'
    elif '0' in x and '1' in x:
        label = 'both_present'
    elif 'Error_with_API_CYKW' in x:
        label = 'error'
    else:
        label = 'neither'
    
    return label

def process_chatgpt_output(df, suffix):
    print('Saving temp df')
    import csv
    #Save temp df:
    df.to_csv('temp_chatgpt_output.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    print('retrieving content')
    df['response_content' + suffix] =  df['response_json' + suffix].apply(lambda x: retrieve_content_from_response_json2(x))
    print('retrieving label')
    df['label' + suffix] = df['response_content' + suffix].apply(lambda x: retrieve_label(x))
    print('Saving temp df2')
    #Save temp df2
    df.to_csv('temp_chatgpt_output2' +suffix + '.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    return df

def combined_lists_to_df(df, encounter_id_list, prompt_list, response_list, prompt, suffix):
    output_df = pd.DataFrame([encounter_id_list, prompt_list, response_list]).transpose()
    output_df = output_df.rename(columns = {0:'encounterkey', 1:prompt, 2:'response_json' + suffix})
    
    df = df.merge(output_df, left_on = ['encounterkey', prompt], right_on = ['encounterkey', prompt], how = 'left')
    df = df[df['response_json' + suffix].notnull()]
    df = process_chatgpt_output(df, suffix)
    return df 


# #### Balanced n = 10000 datasets
import pandas as pd
df_admit_10000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_admit_10000.csv', index_col = 0)
df_radiology_10000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_radiology_10000.csv', index_col = 0)
df_antibiotics_10000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_antibiotics_10000.csv', index_col = 0)


print('admit:', df_admit_10000['history_examination_text_tokens'].sum()/1000*0.002*4)
print('radiology:', df_radiology_10000['history_examination_text_tokens'].sum()/1000*0.002*4)
print('antibiotics:', df_antibiotics_10000['history_examination_text_tokens'].sum()/1000*0.002*4)


df_dict = {
    'df_admit_10000':df_admit_10000,
    'df_radiology_10000':df_radiology_10000,
    'df_antibiotics_10000':df_antibiotics_10000
}

for df_name, df_to_run in df_dict.items():    
    prompt_response_dict = {'prompt_history_examination1':'_history_examination1', 
                            'prompt_history_examination2':'_history_examination2', 
                            'prompt_history_examination3':'_history_examination3', 
                            'prompt_history_examination4':'_history_examination4'}

    for prompt, suffix in prompt_response_dict.items():
        print(df_to_run.shape)
        
        encounter_id_list = []
        prompt_list = []
        response_list = []

        for key, value in dict(zip(df_to_run['encounterkey'].tolist(), df_to_run[prompt].tolist())).items():
            print(key)
            print(len(response_list))
            encounter_id_list.append(key)
            prompt_list.append(value)
            response_list.append(run_chatgpt_api(value))

        df_to_run = combined_lists_to_df(df_to_run, encounter_id_list, prompt_list, response_list, prompt, suffix)

    df_to_save = df_to_run.copy()

    df_to_save.to_csv('path/chatgpt_ed_admit_adults_' + df_name + '_results.csv')


# #### Unbalanced n = 1000 dataset
import pandas as pd
df_admit_unbalanced_1000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_admit_unbalanced_1000.csv', index_col = 0)
df_radiology_unbalanced_1000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_radiology_unbalanced_1000.csv', index_col = 0)
df_antibiotics_unbalanced_1000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_antibiotics_unbalanced_1000.csv', index_col = 0)

df_dict = {
    'df_admit_unbalanced_1000':df_admit_unbalanced_1000,
    'df_radiology_unbalanced_1000':df_radiology_unbalanced_1000,
    'df_antibiotics_unbalanced_1000':df_antibiotics_unbalanced_1000,
}

for df_name, df_to_run in df_dict.items():
    #df_to_run = df_to_run.head(3)
    
    prompt_response_dict = {'prompt_history_examination1':'_history_examination1', 
                            'prompt_history_examination2':'_history_examination2', 
                            'prompt_history_examination3':'_history_examination3', 
                            'prompt_history_examination4':'_history_examination4'}

    for prompt, suffix in prompt_response_dict.items():
        print(df_to_run.shape)

        encounter_id_list = []
        prompt_list = []
        response_list = []

        for key, value in dict(zip(df_to_run['encounterkey'].tolist(), df_to_run[prompt].tolist())).items():
            print(key)
            print(len(response_list))
            encounter_id_list.append(key)
            prompt_list.append(value)
            response_list.append(run_chatgpt_api(value))

        df_to_run = combined_lists_to_df(df_to_run, encounter_id_list, prompt_list, response_list, prompt, suffix)

    df_to_save = df_to_run.copy()

    df_to_save.to_csv('path/chatgpt_ed_admit_adults_' + df_name + '_results.csv')


# #### Sensitivity analysis: balanced n=200 sample

import pandas as pd
df_admit_10000_fixed_200_sample_reversed = pd.read_csv('path/reversed_balanced/df_admit_10000_fixed_200_sample_reversed.csv', index_col = 0)
df_radiology_10000_fixed_200_sample_reversed = pd.read_csv('path/reversed_balanced/df_radiology_10000_fixed_200_sample_reversed.csv', index_col = 0)
df_antibiotics_10000_fixed_200_sample_reversed = pd.read_csv('path/reversed_balanced/df_antibiotics_10000_fixed_200_sample_reversed.csv', index_col = 0)


df_dict = {
    'df_admit_10000_fixed_200_sample_reversed':df_admit_10000_fixed_200_sample_reversed,
    'df_radiology_10000_fixed_200_sample_reversed':df_radiology_10000_fixed_200_sample_reversed,
    'df_antibiotics_10000_fixed_200_sample_reversed':df_antibiotics_10000_fixed_200_sample_reversed
}

for df_name, df_to_run in df_dict.items():    
    prompt_response_dict = {'prompt_history_examination1':'_history_examination1', 
                            'prompt_history_examination2':'_history_examination2', 
                            'prompt_history_examination3':'_history_examination3', 
                            'prompt_history_examination4':'_history_examination4'}

    for prompt, suffix in prompt_response_dict.items():
        print(df_to_run.shape)
        
        encounter_id_list = []
        prompt_list = []
        response_list = []

        for key, value in dict(zip(df_to_run['encounterkey'].tolist(), df_to_run[prompt].tolist())).items():
            print(key)
            print(len(response_list))
            encounter_id_list.append(key)
            prompt_list.append(value)
            response_list.append(run_chatgpt_api(value))

        df_to_run = combined_lists_to_df(df_to_run, encounter_id_list, prompt_list, response_list, prompt, suffix)

    df_to_save = df_to_run.copy()

    df_to_save.to_csv('path/reversed_balanced/chatgpt_ed_admit_adults_' + df_name + '_results.csv')