import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def wilcoxon_signed_rank_test(data_10ipc, data_baseline):
    """
    Performs Wilcoxon Signed-Rank test for each algorithm.

    Parameters:
    data_10ipc (dict): A dictionary with 10ipc case scores.
    data_baseline (dict): A dictionary with baseline case scores.

    Returns:
    dict: A dictionary of Wilcoxon test results (statistics and p-values).
    """
    results = {}
    for key in data_10ipc.keys():
        stat, pvalue = stats.wilcoxon(data_10ipc[key], data_baseline[key])
        results[key] = {"Statistic": stat, "P_Value": pvalue}
    return results

 
def plot_differences_boxplot(data_10ipc, data_baseline):
    """
    Creates a boxplot for the differences between the 10ipc and baseline cases for each algorithm.

    Parameters:
    data_10ipc (dict): A dictionary with 10ipc case scores.
    data_baseline (dict): A dictionary with baseline case scores.
    """
    differences = {key: data_10ipc[key] - data_baseline[key] for key in data_10ipc.keys()}
    plt.boxplot(differences.values(), labels=differences.keys())
    plt.ylabel('Differences (10ipc - Baseline)')
    plt.title('Boxplot of Differences Between Paired Samples')
    plt.show()



def paired_t_test(data_10ipc, data_baseline):
    """
    Performs paired t-tests between the 10ipc and baseline cases for each algorithm.

    Parameters:
    data_10ipc (dict): A dictionary with 10ipc case scores.
    data_baseline (dict): A dictionary with baseline case scores.

    Returns:
    dict: A dictionary of paired t-test results (statistics and p-values).
    """
    results = {}
    for key in data_10ipc.keys():
        stat, pvalue = stats.ttest_rel(data_10ipc[key], data_baseline[key])
        results[key] = {"Statistic": stat, "P_Value": pvalue}
    return results

def bonferroni_correction(t_test_results, num_tests):
    """
    Applies the Bonferroni correction to a dictionary of t-test results.

    Parameters:
    t_test_results (dict): A dictionary where keys are algorithm names and values are dictionaries containing 'P_Value'.
    num_tests (int): The number of tests performed.

    Returns:
    dict: A dictionary with adjusted p-values and a flag indicating significance at alpha = 0.05.
    """
    bonferroni_alpha = 0.05 / num_tests
    bonferroni_adjusted_results = {}
    for algorithm, result in t_test_results.items():
        adjusted_p_value = result['P_Value'] * num_tests
        bonferroni_adjusted_results[algorithm] = {
            'Adjusted P-Value': adjusted_p_value,
            'Significant at 0.05': adjusted_p_value < bonferroni_alpha
        }
    return bonferroni_adjusted_results


def shapiro_wilk_test(data):
    """
    Performs Shapiro-Wilk test for normality on each group in the data.
    
    Parameters:
    data (dict): A dictionary where keys are algorithm names and values are arrays of scores.
    
    Returns:
    dict: A dictionary of Shapiro-Wilk test results (statistics and p-values).
    """
    results = {}
    for key, values in data.items():
        stat, pvalue = stats.shapiro(values)
        #print("HEEEELOO")
        #print(values)
        results[key] = {"Statistic": stat, "P_Value": pvalue}
    return results

def shapiro_wilk_on_differences(data1, data2):
    """
    Performs Shapiro-Wilk test for normality on the differences between pairs of scores from two datasets.

    Parameters:
    data1 (dict): A dictionary where keys are algorithm names and values are arrays of scores for the first condition.
    data2 (dict): A dictionary where keys are algorithm names and values are arrays of scores for the second condition.

    Returns:
    dict: A dictionary of Shapiro-Wilk test results (statistics and p-values) for the differences.
    """
    results = {}
    for key in data1.keys():
        differences = data1[key] - data2[key]
        stat, pvalue = stats.shapiro(differences)
        results[key] = {"Statistic": stat, "P_Value": pvalue}
    return results

def levene_test(data):
    """
    Performs Levene's test for equality of variances among groups.
    
    Parameters:
    data (dict): A dictionary where keys are algorithm names and values are arrays of scores.
    
    Returns:
    dict: Levene's test result (statistic and p-value).
    """
    groups = [values for values in data.values()]
    stat, pvalue = stats.levene(*groups)
    return {"Statistic": stat, "P_Value": pvalue}

def anova_test(data):
    """
    Performs one-way ANOVA test for mean differences among groups.
    
    Parameters:
    data (dict): A dictionary where keys are algorithm names and values are arrays of scores.
    
    Returns:
    dict: ANOVA test result (F-statistic and p-value).
    """
    groups = [values for values in data.values()]
    f_stat, pvalue = stats.f_oneway(*groups)
    return {"F_Statistic": f_stat, "P_Value": pvalue}

def main():
    # check that the numbers are correct
    data_10ipc = {
        "DANN": np.array([79.99999895858765, 78.24999887275696, 78.49999884033203, 78.04999884414673, 78.39999885559082]),
        "MCD": np.array([38.0499987449646, 39.699998693466185, 39.79999867630005, 38.89999875640869, 38.149998752593994]),
        "JAN": np.array([72.4499987258911, 71.04999870491028, 71.69999866485595, 71.94999864006043, 72.04999864387513]),
        "MCC": np.array([60.34999863243103, 56.74999863815307, 59.699998544692995, 57.89999854469299, 57.54999849891663]),
        "CDAN": np.array([79.39999896621704, 78.84999883842468, 80.6999988937378, 79.39999891281128, 78.89999889755249])
    }
    data_baseline = {
    "DANN": np.array([69.0, 67.15, 67.8, 66.8, 66.8]),
    "MCD": np.array([95.6, 96.05, 95.35, 96.2, 94.1]),
    "JAN": np.array([55.0, 50.4, 51.5, 52.95, 49.6]),
    "MCC": np.array([77.75, 76.75, 78.15, 77.0, 74.6]),
    "CDAN": np.array([93.25, 93.05, 93.85, 94.15, 92.45])
    }
    
    a_dist_10ipc = {
    "DANN": np.array([0.0723, 0.1615, 0.2720, 0.4618, 0.6705]),
    "MCD": np.array([2.0, 2.0, 1.9997, 1.9998, 1.9998]),
    "JAN": np.array([1.0913, 0.1915, 0.3950, 0.1950, 0.6907]),
    "MCC": np.array([1.0, 0.8, 1.2, 0.2998, 1.0]),
    "CDAN": np.array([0.4850, 0.0847, 0.1875, 0.8833, 0.5858])
    }

    a_dist_baseline = {
    "DANN": np.array([1.7992, 1.7810, 1.7995, 1.7813, 1.7925]),
    "MCD": np.array([1.6458, 1.6458, 1.6458, 1.6458, 1.6458]),
    "JAN": np.array([1.6725, 1.6691, 1.6643, 1.6418, 1.6728]),
    "MCC": np.array([1.9997, 1.9997, 2.0, 1.9991, 1.9997]),
    "CDAN": np.array([1.7014, 1.6980, 1.6962, 1.7032, 1.6989])
    }
   

    #Experiment:
    # ANOVA to assert statistical difference between groups (algorithms respective score), i.e. eval scores and A-dist.
    # To do ANOVA we need to test equality of variance (valid assumption for in group (compare alg perf within 10ipc or basleine) compariosn).


    print("EVAL SHAPIRO AND ANOVA EXPERIMENT")
    shapiro_10ipc_eval= shapiro_wilk_test(data_10ipc)
    shapiro_baseline_eval= shapiro_wilk_test(data_baseline)

    levene_10ipc_eval = levene_test(data_10ipc)
    levene_baseline_eval = levene_test(data_baseline)
    
    anova_10ipc_eval = anova_test(data_10ipc)
    anova_baseline_eval = anova_test(data_baseline)
    print("Shapiro 10ipc eval: ",shapiro_10ipc_eval)
    print("Shapiro baseline eval: ",shapiro_baseline_eval)
    print("Levene test 10ipc eval: ",levene_10ipc_eval)
    print("Levene test baseline eval", levene_baseline_eval)
    print("anova 10ipc eval: ", anova_10ipc_eval)
    print("anova baseline eval: ", anova_baseline_eval)

    print("A-DIST SHAPRIO AND ANOVA EXPERIMENT")
    shapiro_10ipc_a_dist= shapiro_wilk_test(a_dist_10ipc)
    #print("TESTTTTTTTTT")
    #print(a_dist_10ipc)

    shapiro_baseline_a_dist= shapiro_wilk_test(a_dist_baseline)

    levene_10ipc_a_dist = levene_test(a_dist_10ipc)
    levene_baseline_a_dist = levene_test(a_dist_baseline)
    
    anova_10ipc_a_dist = anova_test(a_dist_10ipc)
    anova_baseline_a_dist = anova_test(a_dist_baseline)
    print("Shapiro 10ipc a_dist: ",shapiro_10ipc_a_dist)
    print("Shapiro baseline a_dist: ",shapiro_baseline_a_dist)
    print("Levene test 10ipc a_dist: ",levene_10ipc_a_dist)
    print("Levene test baseline a_dist", levene_baseline_a_dist)
    print("anova 10ipc a_dist: ", anova_10ipc_a_dist)
    print("anova baseline a_dist: ", anova_baseline_a_dist)


    print("EVAL: Paired t-test between the same algortims, comparing algorithm X between baseline and 10ipc")
    #Shaprio WILK on the differences between 10ipc and baseline
    #Then do paired t-tests (homogenairty of variances is not needed to be assesed for paired t-tests, only independant t-tests)
    eval_shapiro_diff = shapiro_wilk_on_differences(data_10ipc,data_baseline)
    paired_t_test_eval = paired_t_test(data_10ipc,data_baseline)
    corrected_paired_t_test_eval = bonferroni_correction(paired_t_test_eval,5)
    print("eval_shapiro_diff: ",eval_shapiro_diff)
    print("paired_t_test_eval: ", paired_t_test_eval)
    print("Bonferroni corrected paired t-test eval: ",corrected_paired_t_test_eval)


    print("ADIST: Paired t-test between the same algortims, comparing algorithm X between baseline and 10ipc")
    #Shaprio WILK on the differences between 10ipc and baseline
    #Then do paired t-tests (homogenairty of variances is not needed to be assesed for paired t-tests, only independant t-tests)
    a_dist_shapiro_diff = shapiro_wilk_on_differences(a_dist_10ipc,a_dist_baseline)
    a_dist_shapiro_diff = shapiro_wilk_on_differences(a_dist_10ipc,a_dist_baseline)
    paired_t_test_a_dist = paired_t_test(a_dist_10ipc,a_dist_baseline)
    corrected_paired_t_test_a_dist = bonferroni_correction(paired_t_test_a_dist,5)
    print("a_dist_shapiro_diff: ",a_dist_shapiro_diff)
    print("paired_t_test_a_dist: ", paired_t_test_a_dist)
    print("Bonferroni corrected paired t-test eval: ",corrected_paired_t_test_a_dist)





if __name__ == "__main__":
    main()