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
        stat, pvalue = stats.wilcoxon(data_10ipc[key], data_baseline[key + "_baseline"])
        results[key] = {"Statistic": stat, "P_Value": pvalue}
    return results

 
def plot_differences_boxplot(data_10ipc, data_baseline):
    """
    Creates a boxplot for the differences between the 10ipc and baseline cases for each algorithm.

    Parameters:
    data_10ipc (dict): A dictionary with 10ipc case scores.
    data_baseline (dict): A dictionary with baseline case scores.
    """
    differences = {key: data_10ipc[key] - data_baseline[key + "_baseline"] for key in data_10ipc.keys()}
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
        stat, pvalue = stats.ttest_rel(data_10ipc[key], data_baseline[key + "_baseline"])
        results[key] = {"Statistic": stat, "P_Value": pvalue}
    return results

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
        "DANN_10ipc": np.array([79.99999895858765, 78.24999887275696, 78.49999884033203, 78.04999884414673, 78.39999885559082]),
        "MCD_10ipc": np.array([38.0499987449646, 39.699998693466185, 39.79999867630005, 38.89999875640869, 38.149998752593994]),
        "JAN_10ipc": np.array([72.4499987258911, 71.04999870491028, 71.69999866485595, 71.94999864006043, 72.04999864387513]),
        "MCC_10ipc": np.array([60.34999863243103, 56.74999863815307, 59.699998544692995, 57.89999854469299, 57.54999849891663]),
        "CDAN_10ipc": np.array([79.39999896621704, 78.84999883842468, 80.6999988937378, 79.39999891281128, 78.89999889755249])
    }
    data_baseline = {
    "DANN_baseline": np.array([69.0, 67.15, 67.8, 66.8, 66.8]),
    "MCD_baseline": np.array([95.6, 96.05, 95.35, 96.2, 94.1]),
    "JAN_baseline": np.array([55.0, 50.4, 51.5, 52.95, 49.6]),
    "MCC_baseline": np.array([77.75, 76.75, 78.15, 77.0, 74.6]),
    "CDAN_baseline": np.array([93.25, 93.05, 93.85, 94.15, 92.45])
}
    
    #Add maps for a_dist_10ipc and a_dist_baseline

    # Conduct tests
    shapiro_results = shapiro_wilk_test(data_10ipc)
    levene_results = levene_test(data_10ipc)
    anova_results = anova_test(data_10ipc)
    paired_t_results = paired_t_test(data_10ipc, data_baseline)

    #Change to use A-distance data instead
    wilcoxon_results = wilcoxon_signed_rank_test(data_10ipc, data_baseline)

    # Display results
    print("Shapiro-Wilk Test Results:", shapiro_results)
    print("Levene's Test Results:", levene_results)
    print("ANOVA Test Results:", anova_results)

if __name__ == "__main__":
    main()