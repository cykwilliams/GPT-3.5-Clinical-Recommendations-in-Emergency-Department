# ## Main analysis
import pandas as pd
df_admit_10000_fixed = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_admit_1000_results.csv', index_col = 0)
df_radiology_10000_fixed = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_radiology_1000_results.csv', index_col = 0)
df_antibiotics_10000_fixed = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_antibiotics_1000_results.csv', index_col = 0)


# #### Get word counts for n=10000 dfs for Supplement
df_admit_10000_fixed.columns
df_names = ['admit', 'radiology', 'antibiotics']
df_list = [df_admit_10000_fixed, df_radiology_10000_fixed, df_antibiotics_10000_fixed]

df_dict = dict(zip(df_names, df_list))

task_list = []
prompt_list = []
mean_word_count_list = []
std_word_count_list = []

for df_name, df in df_dict.items():
    for response in ['response_content_history_examination1', 'response_content_history_examination2', 'response_content_history_examination3', 'response_content_history_examination4']:
        mean_word_count = df[response].apply(lambda x: len(str(x).split())).mean()
        std_word_count = df[response].apply(lambda x: len(str(x).split())).std()
        task_list.append(df_name)
        prompt_list.append(response)
        mean_word_count_list.append(round(mean_word_count, 1))
        std_word_count_list.append(round(std_word_count, 1))

df_mean_word_count_10000 = pd.DataFrame({
    'task': task_list,
    'prompt': prompt_list,
    'mean_word_count': mean_word_count_list,
    'std_word_count': std_word_count_list
})

#Save to csv
df_mean_word_count_10000.to_csv('path/df_mean_word_count_10000.csv')
#Display
df_mean_word_count_10000


# #### Examine confusion matrix
##Make list of outcomes so can save to csv:

def get_eval_metrics_main(df, outcome):
    outcome_list = []
    label_list = []
    sensitivity_list = []
    specificity_list = []
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    f1_binary_list = []
    f1_micro_list = []
    f1_macro_list = []
    f1_weighted_list = []

    for label in ['label_history_examination1', 'label_history_examination2', 'label_history_examination3', 'label_history_examination4']:
        confusion_matrix_admit = pd.crosstab(df[outcome], df[label], rownames=[outcome], colnames=[label], margins=True)

        # Calculate TP, FP, TN, FN
        TP = confusion_matrix_admit[1][1]
        FP = confusion_matrix_admit[1][0]
        TN = confusion_matrix_admit[0][0]
        FN = confusion_matrix_admit[0][1]

        # Print the confusion matrix
        print(confusion_matrix_admit)
        print("True Positives (TP):", TP)
        print("False Positives (FP):", FP)
        print("True Negatives (TN):", TN)
        print("False Negatives (FN):", FN)


        # Calculate sensitivity (true positive rate)
        sensitivity = TP / (TP + FN)

        # Calculate specificity (true negative rate)
        specificity = TN / (TN + FP)

        # Print the results
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity, '\n')

        ##Calculate F1 score - need to remove all the errors here first
        from sklearn.metrics import f1_score
        # Calculate the F1 score
        f1_binary = f1_score(df[outcome], df[label], average = 'binary')
        # Print the result
        print("F1 score (binary):", f1_binary)
        
        f1_micro = f1_score(df[outcome], df[label], average = 'micro')
        # Print the result
        print("F1 score (micro):", f1_micro)
        
        f1_macro = f1_score(df[outcome], df[label], average = 'macro')
        # Print the result
        print("F1 score (macro):", f1_macro)
        
        f1_weighted = f1_score(df[outcome], df[label], average = 'weighted')
        # Print the result
        print("F1 score (weighted):", f1_weighted, '\n\n')
        
        #Append to lists:
        outcome_list.append(outcome)
        label_list.append(label)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_binary_list.append(f1_binary)
        f1_micro_list.append(f1_micro)
        f1_macro_list.append(f1_macro)
        f1_weighted_list.append(f1_weighted)
        
    summary_results = pd.DataFrame([outcome_list, label_list, TP_list, FP_list, TN_list, FN_list, sensitivity_list, specificity_list, f1_binary_list, f1_micro_list, f1_macro_list, f1_weighted_list]).transpose()
    summary_results.columns = ['Task', 'Prompt', 'TP', 'FP', 'TN', 'FN', 'Sensitivity', 'Specificity', 'F1_binary', 'F1_micro', 'F1_macro', 'F1_weighted']    
    
    return summary_results

summary_results_admit_10000 = get_eval_metrics_main(df_admit_10000_fixed, 'admitted')
summary_results_radiology_10000 = get_eval_metrics_main(df_radiology_10000_fixed, 'imaging_orders_binary')
summary_results_antibiotics_10000 = get_eval_metrics_main(df_antibiotics_10000_fixed, 'abx_ordered_ed')

summary_results_10000 = pd.concat([summary_results_admit_10000, summary_results_radiology_10000, summary_results_antibiotics_10000])
summary_results_10000

###Now compare GPT vs physician labels
# ## Import labelled n=200 samples and merge with master df
import pandas as pd
df_admit_10000_fixed_200_sample_classified = pd.read_csv('path/df_admit_10000_fixed_200_sample_classified.csv', index_col = 0)
df_radiology_10000_fixed_200_sample_classified = pd.read_csv('path/df_radiology_10000_fixed_200_sample_classified.csv', index_col = 0)
df_antibiotics_10000_fixed_200_sample_classified = pd.read_csv('path/df_antibiotics_10000_fixed_200_sample_classified.csv', index_col = 0)

df_admit_10000_fixed_200_sample_classified['CYKW_admitted'] = df_admit_10000_fixed_200_sample_classified['CYKW_admitted'].map({'Y': 1, 'N': 0})
df_admit_10000_fixed_200_sample = df_admit_10000_fixed.merge(df_admit_10000_fixed_200_sample_classified[['patientdurablekey', 'edvisitkey', 'encounterkey', 'CYKW_admitted']], on = ['patientdurablekey', 'edvisitkey', 'encounterkey'], how = 'right')

df_radiology_10000_fixed_200_sample_classified['CYKW_imaging_orders_binary'] = df_radiology_10000_fixed_200_sample_classified['CYKW_imaging_orders'].map({'N':0, 'XR':1, 'CT':1, 'US':1, 'MRI':1})
df_radiology_10000_fixed_200_sample = df_radiology_10000_fixed.merge(df_radiology_10000_fixed_200_sample_classified[['patientdurablekey', 'edvisitkey', 'encounterkey', 'CYKW_imaging_orders_binary']], on = ['patientdurablekey', 'edvisitkey', 'encounterkey'], how = 'right')

df_antibiotics_10000_fixed_200_sample_classified['CYKW_abx_ordered_ed'] = df_antibiotics_10000_fixed_200_sample_classified['CYKW_abx_ordered_ed'].map({'Y': 1, 'N': 0})
df_antibiotics_10000_fixed_200_sample = df_antibiotics_10000_fixed.merge(df_antibiotics_10000_fixed_200_sample_classified[['patientdurablekey', 'edvisitkey', 'encounterkey', 'CYKW_abx_ordered_ed']], on = ['patientdurablekey', 'edvisitkey', 'encounterkey'], how = 'right')

##Make list of outcomes so can save to csv:

def get_eval_metrics_main(df, outcome):
    outcome_list = []
    label_list = []
    sensitivity_list = []
    specificity_list = []
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    f1_binary_list = []
    f1_micro_list = []
    f1_macro_list = []
    f1_weighted_list = []

    for label in ['label_history_examination1', 'label_history_examination2', 'label_history_examination3', 'label_history_examination4']:
        confusion_matrix_admit = pd.crosstab(df[outcome], df[label], rownames=[outcome], colnames=[label], margins=True)

        # Calculate TP, FP, TN, FN
        TP = confusion_matrix_admit[1][1]
        FP = confusion_matrix_admit[1][0]
        TN = confusion_matrix_admit[0][0]
        FN = confusion_matrix_admit[0][1]

        # Print the confusion matrix
        print(confusion_matrix_admit)
        print("True Positives (TP):", TP)
        print("False Positives (FP):", FP)
        print("True Negatives (TN):", TN)
        print("False Negatives (FN):", FN)


        # Calculate sensitivity (true positive rate)
        sensitivity = TP / (TP + FN)

        # Calculate specificity (true negative rate)
        specificity = TN / (TN + FP)

        # Print the results
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity, '\n')

        ##Calculate F1 score - need to remove all the errors here first
        from sklearn.metrics import f1_score
        # Calculate the F1 score
        f1_binary = f1_score(df[outcome], df[label], average = 'binary')
        # Print the result
        print("F1 score (binary):", f1_binary)
        
        f1_micro = f1_score(df[outcome], df[label], average = 'micro')
        # Print the result
        print("F1 score (micro):", f1_micro)
        
        f1_macro = f1_score(df[outcome], df[label], average = 'macro')
        # Print the result
        print("F1 score (macro):", f1_macro)
        
        f1_weighted = f1_score(df[outcome], df[label], average = 'weighted')
        # Print the result
        print("F1 score (weighted):", f1_weighted, '\n\n')
        
        #Append to lists:
        outcome_list.append(outcome)
        label_list.append(label)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_binary_list.append(f1_binary)
        f1_micro_list.append(f1_micro)
        f1_macro_list.append(f1_macro)
        f1_weighted_list.append(f1_weighted)
        
    summary_results = pd.DataFrame([outcome_list, label_list, TP_list, FP_list, TN_list, FN_list, sensitivity_list, specificity_list, f1_binary_list, f1_micro_list, f1_macro_list, f1_weighted_list]).transpose()
    summary_results.columns = ['Task', 'Prompt', 'TP', 'FP', 'TN', 'FN', 'Sensitivity', 'Specificity', 'F1_binary', 'F1_micro', 'F1_macro', 'F1_weighted']    
    
    return summary_results

def get_eval_metrics(df, variable, CYKW_variable):
    outcome_list = []
    label_list = []
    sensitivity_list = []
    specificity_list = []
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    f1_binary_list = []
    f1_micro_list = []
    f1_macro_list = []
    f1_weighted_list = []
    
    print('CYKW stats:')
    confusion_matrix_admit = pd.crosstab(df[variable], df[CYKW_variable], rownames=[variable], colnames=[CYKW_variable], margins=True)

    # Calculate TP, FP, TN, FN
    TP = confusion_matrix_admit[1][1]
    FP = confusion_matrix_admit[1][0]
    TN = confusion_matrix_admit[0][0]
    FN = confusion_matrix_admit[0][1]

    # Print the confusion matrix
    print(confusion_matrix_admit)
    print("True Positives (TP):", TP)
    print("False Positives (FP):", FP)
    print("True Negatives (TN):", TN)
    print("False Negatives (FN):", FN)
    
    
    # Calculate sensitivity (true positive rate)
    sensitivity = TP / (TP + FN)

    # Calculate specificity (true negative rate)
    specificity = TN / (TN + FP)

    # Print the results
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity, '\n')
    
    ##Calculate F1 score - need to remove all the errors here first
    from sklearn.metrics import f1_score
    f1_binary = f1_score(df[variable], df[CYKW_variable], average = 'binary')
    # Print the result
    print("F1 score (binary):", f1_binary)

    f1_micro = f1_score(df[variable], df[CYKW_variable], average = 'micro')
    # Print the result
    print("F1 score (micro):", f1_micro)

    f1_macro = f1_score(df[variable], df[CYKW_variable], average = 'macro')
    # Print the result
    print("F1 score (macro):", f1_macro)

    f1_weighted = f1_score(df[variable], df[CYKW_variable], average = 'weighted')
    # Print the result
    print("F1 score (weighted):", f1_weighted, '\n\n')
    
    outcome_list.append(variable)
    label_list.append('CYKW')
    TP_list.append(TP)
    FP_list.append(FP)
    TN_list.append(TN)
    FN_list.append(FN)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    f1_binary_list.append(f1_binary)
    f1_micro_list.append(f1_micro)
    f1_macro_list.append(f1_macro)
    f1_weighted_list.append(f1_weighted)
    
    print('ChatGPT stats:')
    for label in ['label_history_examination1', 'label_history_examination2', 'label_history_examination3', 'label_history_examination4']:
        confusion_matrix_admit = pd.crosstab(df[variable], df[label], rownames=[variable], colnames=[label], margins=True)

        # Calculate TP, FP, TN, FN
        TP = confusion_matrix_admit[1][1]
        FP = confusion_matrix_admit[1][0]
        TN = confusion_matrix_admit[0][0]
        FN = confusion_matrix_admit[0][1]

        # Print the confusion matrix
        print(confusion_matrix_admit)
        print("True Positives (TP):", TP)
        print("False Positives (FP):", FP)
        print("True Negatives (TN):", TN)
        print("False Negatives (FN):", FN)
        
        
        # Calculate sensitivity (true positive rate)
        sensitivity = TP / (TP + FN)

        # Calculate specificity (true negative rate)
        specificity = TN / (TN + FP)

        # Print the results
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity, '\n')
        
        ##Calculate F1 score - need to remove all the errors here first
        from sklearn.metrics import f1_score
        # Calculate the F1 score
        f1_binary = f1_score(df[variable], df[label], average = 'binary')
        # Print the result
        print("F1 score (binary):", f1_binary)
        
        f1_micro = f1_score(df[variable], df[label], average = 'micro')
        # Print the result
        print("F1 score (micro):", f1_micro)
        
        f1_macro = f1_score(df[variable], df[label], average = 'macro')
        # Print the result
        print("F1 score (macro):", f1_macro)
        
        f1_weighted = f1_score(df[variable], df[label], average = 'weighted')
        # Print the result
        print("F1 score (weighted):", f1_weighted, '\n\n')
        
        outcome_list.append(variable)
        label_list.append(label)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_binary_list.append(f1_binary)
        f1_micro_list.append(f1_micro)
        f1_macro_list.append(f1_macro)
        f1_weighted_list.append(f1_weighted)

    summary_results = pd.DataFrame([outcome_list, label_list, TP_list, FP_list, TN_list, FN_list, sensitivity_list, specificity_list, f1_binary_list, f1_micro_list, f1_macro_list, f1_weighted_list]).transpose()
    summary_results.columns = ['Task', 'Prompt', 'TP', 'FP', 'TN', 'FN', 'Sensitivity', 'Specificity', 'F1_binary', 'F1_micro', 'F1_macro', 'F1_weighted']    
    
    return summary_results


summary_results_admit_10000_fixed_200_sample = get_eval_metrics(df_admit_10000_fixed_200_sample, 'admitted', 'CYKW_admitted')
summary_results_radiology_10000_fixed_200_sample = get_eval_metrics(df_radiology_10000_fixed_200_sample, 'imaging_orders_binary', 'CYKW_imaging_orders_binary')
summary_results_antibiotics_10000_fixed_200_sample = get_eval_metrics(df_antibiotics_10000_fixed_200_sample, 'abx_ordered_ed', 'CYKW_abx_ordered_ed')

summary_results_200 = pd.concat([summary_results_admit_10000_fixed_200_sample, summary_results_radiology_10000_fixed_200_sample, summary_results_antibiotics_10000_fixed_200_sample])


##Merge the main analysis and n=200 subsample:
#First, label sample size in separate column
summary_results_10000['Sample'] = '10000'
summary_results_200['Sample'] = '200'

summary_results_main = pd.concat([summary_results_10000, summary_results_200])

###Save final results
summary_results_main.to_csv('path/chatgpt_ed_admit_adults_290523_results_summary_main.csv')

