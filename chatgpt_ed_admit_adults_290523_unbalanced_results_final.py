import pandas as pd
df_admit_unbalanced_1000_fixed = pd.read_csv('path/chatgpt_ed_admit_adults_df_admit_unbalanced_1000_results.csv', index_col = 0)
df_radiology_unbalanced_1000_fixed = pd.read_csv('path/chatgpt_ed_admit_adults_df_radiology_unbalanced_1000_results.csv', index_col = 0)
df_antibiotics_unbalanced_1000_fixed = pd.read_csv('path/chatgpt_ed_admit_adults_df_antibiotics_unbalanced_1000_results.csv', index_col = 0)
#Import the labelled dataset
df_CYKW_unbalanced_1000 = pd.read_csv('path/chatgpt_ed_admit_adults_master_processed_df_CYKW_classified_unbalanced_1000.csv', index_col = 0)

##Confirm the samples match with each other:
def are_lists_same(list1, list2, list3, list4):
    return print(list1 == list2 == list3 == list4)

are_lists_same(df_admit_unbalanced_1000_fixed.encounterkey.tolist(),
              df_radiology_unbalanced_1000_fixed.encounterkey.tolist(),
              df_antibiotics_unbalanced_1000_fixed.encounterkey.tolist(),
              df_CYKW_unbalanced_1000.encounterkey.tolist())
are_lists_same(df_admit_unbalanced_1000_fixed.edvisitkey.tolist(),
              df_radiology_unbalanced_1000_fixed.edvisitkey.tolist(),
              df_antibiotics_unbalanced_1000_fixed.edvisitkey.tolist(),
              df_CYKW_unbalanced_1000.edvisitkey.tolist())


for column in ['admitted', 'label_history_examination1', 'label_history_examination2', 'label_history_examination3', 'label_history_examination4']:
    print(df_admit_unbalanced_1000_fixed[column].value_counts(), '\n')
for column in ['imaging_orders_binary', 'label_history_examination1', 'label_history_examination2', 'label_history_examination3', 'label_history_examination4']:
    print(df_radiology_unbalanced_1000_fixed[column].value_counts(), '\n')
for column in ['abx_ordered_ed', 'label_history_examination1', 'label_history_examination2', 'label_history_examination3', 'label_history_examination4']:
    print(df_antibiotics_unbalanced_1000_fixed[column].value_counts(), '\n')


# ## Eval metrics for n=1000 unbalanced sample
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

##Merge CYKW label to original dfs
df_admit_unbalanced_1000_fixed = df_admit_unbalanced_1000_fixed.merge(df_CYKW_unbalanced_1000[['encounterkey', 'CYKW_admit']], on = ['encounterkey'], how = 'left')
df_radiology_unbalanced_1000_fixed = df_radiology_unbalanced_1000_fixed.merge(df_CYKW_unbalanced_1000[['encounterkey', 'CYKW_imaging_orders_binary']], on = ['encounterkey'], how = 'left')
df_antibiotics_unbalanced_1000_fixed = df_antibiotics_unbalanced_1000_fixed.merge(df_CYKW_unbalanced_1000[['encounterkey', 'CYKW_abx_ordered_ed']], on = ['encounterkey'], how = 'left')

##Replace Y/N with 1/0:
df_admit_unbalanced_1000_fixed['CYKW_admit'] = df_admit_unbalanced_1000_fixed['CYKW_admit'].replace({'Y':1, 'N':0})
df_radiology_unbalanced_1000_fixed['CYKW_imaging_orders_binary'] = df_radiology_unbalanced_1000_fixed['CYKW_imaging_orders_binary'].replace({'Y':1, 'N':0})
df_antibiotics_unbalanced_1000_fixed['CYKW_abx_ordered_ed'] = df_antibiotics_unbalanced_1000_fixed['CYKW_abx_ordered_ed'].replace({'Y':1, 'N':0})

summary_results_admit_unbalanced_1000_fixed = get_eval_metrics(df_admit_unbalanced_1000_fixed, 'admitted', 'CYKW_admit')

summary_results_radiology_unbalanced_1000_fixed = get_eval_metrics(df_radiology_unbalanced_1000_fixed, 'imaging_orders_binary', 'CYKW_imaging_orders_binary')

summary_results_antibiotics_unbalanced_1000_fixed = get_eval_metrics(df_antibiotics_unbalanced_1000_fixed, 'abx_ordered_ed', 'CYKW_abx_ordered_ed')

summary_results_admit_unbalanced_1000_fixed
summary_results_radiology_unbalanced_1000_fixed
summary_results_antibiotics_unbalanced_1000_fixed

##Concatenate the three tasks into one df
summary_results_unbalanced_1000 = pd.concat([summary_results_antibiotics_unbalanced_1000_fixed, summary_results_radiology_unbalanced_1000_fixed, summary_results_antibiotics_unbalanced_1000_fixed])


summary_results_unbalanced_1000.to_csv('path/chatgpt_ed_admit_adults_290523_results_summary_unbalanced_1000.csv')
