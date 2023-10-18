import pandas as pd
#Import balanced GPT3.5 reversed n=200 sample
df_admit_10000_fixed_200_sample_reversed = pd.read_csv('path/reversed_balanced/chatgpt_ed_admit_adults_df_admit_10000_fixed_200_sample_reversed_results.csv', index_col = 0)
df_radiology_10000_fixed_200_sample_reversed = pd.read_csv('path/reversed_balanced/chatgpt_ed_admit_adults_df_radiology_10000_fixed_200_sample_reversed_results.csv', index_col = 0)
df_antibiotics_10000_fixed_200_sample_reversed = pd.read_csv('path/reversed_balanced/chatgpt_ed_admit_adults_df_antibiotics_10000_fixed_200_sample_reversed_results.csv', index_col = 0)

##Confirm the samples match with the CYKW_200_samples:
import pandas as pd
df_admit_10000_fixed_200_sample_classified = pd.read_csv('path/df_admit_10000_fixed_200_sample_classified.csv', index_col = 0)
df_radiology_10000_fixed_200_sample_classified = pd.read_csv('path/df_radiology_10000_fixed_200_sample_classified.csv', index_col = 0)
df_antibiotics_10000_fixed_200_sample_classified = pd.read_csv('path/df_antibiotics_10000_fixed_200_sample_classified.csv', index_col = 0)

def are_lists_same(list1, list2):
    return print(list1 == list2)

are_lists_same(df_admit_10000_fixed_200_sample_classified.encounterkey.tolist(),
              df_admit_10000_fixed_200_sample_reversed.encounterkey.tolist())
are_lists_same(df_radiology_10000_fixed_200_sample_classified.encounterkey.tolist(),
              df_radiology_10000_fixed_200_sample_reversed.encounterkey.tolist())
are_lists_same(df_antibiotics_10000_fixed_200_sample_classified.encounterkey.tolist(),
              df_antibiotics_10000_fixed_200_sample_reversed.encounterkey.tolist())

#Convert labels to binary
df_admit_10000_fixed_200_sample_classified['CYKW_admitted'] = df_admit_10000_fixed_200_sample_classified['CYKW_admitted'].map({'Y': 1, 'N': 0})
df_radiology_10000_fixed_200_sample_classified['CYKW_imaging_orders_binary'] = df_radiology_10000_fixed_200_sample_classified['CYKW_imaging_orders'].map({'N':0, 'XR':1, 'CT':1, 'US':1, 'MRI':1})
df_antibiotics_10000_fixed_200_sample_classified['CYKW_abx_ordered_ed'] = df_antibiotics_10000_fixed_200_sample_classified['CYKW_abx_ordered_ed'].map({'Y': 1, 'N': 0})

#Add in CYKW_label
df_admit_10000_fixed_200_sample_reversed = df_admit_10000_fixed_200_sample_reversed.merge(df_admit_10000_fixed_200_sample_classified[['encounterkey', 'CYKW_admitted']], on = 'encounterkey', how = 'left')
df_radiology_10000_fixed_200_sample_reversed = df_radiology_10000_fixed_200_sample_reversed.merge(df_radiology_10000_fixed_200_sample_classified[['encounterkey', 'CYKW_imaging_orders_binary']], on = 'encounterkey', how = 'left')
df_antibiotics_10000_fixed_200_sample_reversed = df_antibiotics_10000_fixed_200_sample_reversed.merge(df_antibiotics_10000_fixed_200_sample_classified[['encounterkey', 'CYKW_abx_ordered_ed']], on = 'encounterkey', how = 'left')

# ## Examine results of sensitivity analysis
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

#GPT3.5 reversed:
summary_results_admit_10000_fixed_200_sample_reversed = get_eval_metrics(df_admit_10000_fixed_200_sample_reversed, 'admitted', 'CYKW_admitted')
summary_results_radiology_10000_fixed_200_sample_reversed = get_eval_metrics(df_radiology_10000_fixed_200_sample_reversed, 'imaging_orders_binary', 'CYKW_imaging_orders_binary')
summary_results_antibiotics_10000_fixed_200_sample_reversed = get_eval_metrics(df_antibiotics_10000_fixed_200_sample_reversed, 'abx_ordered_ed', 'CYKW_abx_ordered_ed')

summary_results_200_sensitivity = pd.concat([summary_results_admit_10000_fixed_200_sample_reversed, summary_results_radiology_10000_fixed_200_sample_reversed, summary_results_antibiotics_10000_fixed_200_sample_reversed])
summary_results_200_sensitivity

####Save results
summary_results_200_sensitivity.to_csv('path/chatgpt_ed_admit_adults_290523_results_summary_200_sensitivity.csv')

