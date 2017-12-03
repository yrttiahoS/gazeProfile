import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import lmfit
import math
from lmfit.models import QuadraticModel
from os import listdir
from os.path import isfile, join

##TO DO: modify functions to use df.column.apply(function())?

ids = ['all', 'control', 'neutral', 'happy', 'fearful']
#id_index_order = [1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4] # or [1, 1, 1, 2, 2, 3] and [2, 3, 4, 3, 4, 4] --> this will make sense later
difference_list = ['control_neutral', 'control_happy', 'control_fearful', 'neutral_happy', 'neutral_fearful', 'happy_fearful']
control_directory = ''

def plot_style():
    plt.style.use(['default', 'seaborn-ticks'])
    figure_size = plt.rcParams["figure.figsize"]
    figure_size[0] = 7
    figure_size[1] = 5
    plt.rcParams["figure.figsize"] = figure_size


def get_control_data(data_directory):
    global control_directory
    control_directory = data_directory
    control_srt_subjects, control_srt_trials = control_analysis('SRT')
    control_face_subjects, control_face_trials = control_analysis('Face')

    control_data = pd.merge(control_srt_subjects, control_face_subjects, on='subject', how="right")
    #control_data_trials = pd.merge(control_srt_trials, control_face_trials, on='subject', how="right") # Currently not needed

    return control_data #, control_data_trials


def control_analysis(task_type):
    data = add_all_files(task_type)

    data['subject'] = data.filename.str.partition("_")[0]

    if task_type == 'SRT':
        group_trials, group_subjects = srt_analysis(data)
    elif task_type == 'Face':   
        group_trials, group_subjects = face_analysis(data)

    return group_subjects, group_trials


def add_all_files(task_type):
    data = pd.DataFrame()
    path = sjoin(control_directory, task_type)
    #needs cleanup
    for file in [f for f in listdir(path) if isfile(sjoin(path, file))]:
        data = data.append(pd.read_csv(sjoin(path, file)), ignore_index=True)
    data[data < 0] = np.nan # nan is better than -1
    return data


def srt_analysis(df):
    group_trials, group_subjects, subject_list = analysis_files(df)

    for index, subject in enumerate(subject_list):
        subject_data = pd.DataFrame()
        sub = df[(df.subject == subject)]

        subject_data['subject'] = [subject] * len(subject_data.index)
        subject_data['trial'] = sub.trialnumber ## CHECK IF CORRECT
        subject_data['srt'] = sub.combination
        subject_data[subject_data.srt == 1000] = np.nan # reaction times of 1000 ms are not valid (no gaze shift) --> remove

        group_trials = group_trials.append(subject_data, ignore_index=True)

        subject_data['missing'] = subject_data.srt.isnull().sum()
        subject_data['srt_median'] = subject_data.srt.median()

        subject_data.drop(['srt', 'trial'], axis=1, inplace=True, errors='ignore')
        group_subjects = group_subjects.append(subject_data.head(1), ignore_index=True)

    group_subjects['srt_zscore'] = st.zscore(group_subjects.srt_med)

    return group_trials, group_subjects


def analysis_files(df):
    group_trials = pd.DataFrame()
    group_subjects = pd.DataFrame()
    subject_list = np.unique(df.subject)

    return group_trials, group_subjects, subject_list


def face_analysis(df):
    group_trials, group_subjects, subjects = analysis_files(df)

    for index, subject in enumerate(subjects):
        subject_data = pd.DataFrame()
        sub = df[(df.subject == subject)]

        subject_data['subject'] = [subject] * len(subject_data.index)
        subject_data['stimulus'] = df.condition.str[:-4]

        stimuli = get_stimulustypes(sub)
        filter_list, pfix_list, success_list, shown_list, success_percentage_list, miss_list = get_stimuli_lists()

        for i, f in enumerate(stimuli):
            subject_data[filter_list[i]] = stimuli[i].csaccpindex
            subject_data[pfix_list[i]] = subject_data[filter_list[i]].dropna().mean()
            subject_data[success_list[i]] = stimuli[i][stimuli[i]['technical error'] == 0]['technical error'].count()
            subject_data[shown_list[i]] = stimuli[i]['technical error'].count()
            subject_data[success_percentage_list[i]] = subject_data[success_list[i]] / subject_data[shown_list[i]]
            subject_data[miss_list[i]] = subject_data[shown_list[i]] / subject_data[success_list[i]]

        group_trials = group_trials.append(subject_data.loc[:,['subject', 'stimulus', 'trial_all', 'trial_control', 'trial_neutral', 'trial_happy', 'trial_fearful']], ignore_index=True)
        subject_data.drop(['stimulus', 'trial_all', 'trial_control', 'trial_neutral', 'trial_happy', 'trial_fearful'], axis=1, inplace=True, errors='ignore') # this is a bit copy-paste-y, please fix
        group_subjects = group_subjects.append(subject_data.head(1), ignore_index=True)

    return group_trials, group_subjects


def get_stimuli_lists():
    filters, pfix, success, shown, success_percentage, miss = ([] for x in range(6))

    for i in ids:
        filters.append(ujoin('trial', i))
        pfix.append(ujoin('pfix', i))
        success.append(ujoin('success', i))
        shown.append(ujoin('shown', i))
        success_percentage.append(ujoin('success_percentage', i))
        miss.append(ujoin('miss', i))

    return filters, pfix, success, shown, success_percentage, miss


def get_srt_std(group_trials, data): #rename?
    subject_std = []
    sub_list = data.subject.tolist()
    for s in sub_list:
        sub = group_trials[group_trials.subject == s]
        subject_std.append(np.std(sub.srt.dropna()))
    data['srt_std'] = subject_std
    square_transformed = np.sqrt(data.srt_std)
    square_zscore = st.zscore(square_transformed)
    data['square_srt_std'] = square_transformed
    data['square_srt_std_zscore'] = square_zscore

    print("Shapiro-Wilk test of normality for non-transformed data: " + str(st.shapiro(data.srt_std)))
    print("Shapiro-Wilk test of normality for square root transformed data: " + str(st.shapiro(square_transformed)))

    return data

# Unfinished but currently not needed
#def getCI(group_trials, data):
    #return "this is unfinished"


def add_differences(data):
    # needs cleanup
    data[difference_list[0]] = data.pfix_control - data.pfix_neutral
    data[difference_list[1]] = data.pfix_control - data.pfix_happy
    data[difference_list[2]] = data.pfix_control - data.pfix_fearful
    data[difference_list[3]] = data.pfix_neutral - data.pfix_happy
    data[difference_list[4]] = data.pfix_neutral - data.pfix_fearful
    data[difference_list[5]] = data.pfix_happy - data.pfix_fearful

    for i in range(6):
        data[ujoin(difference_list[i], 'absolute')] = abs(data.difference_list[i])
    
    #difference of weighted values --> needs cleanup (like above, combine these into one)
    data['cvsnc'] = data.comb_c - data.comb_n
    data['cvshc'] = data.comb_c - data.comb_h
    data['cvsfc'] = data.comb_c - data.comb_f
    data['nvshc'] = data.comb_n - data.comb_h
    data['nvsfc'] = data.comb_n - data.comb_f
    data['hvsfc'] = data.comb_h - data.comb_f

    #absolute weighted differences --> needs cleanup
    data['cvsnac'] = abs(data.cvsnc)
    data['cvshac'] = abs(data.cvshc)
    data['cvsfac'] = abs(data.cvsfc)
    data['nvshac'] = abs(data.nvshc)
    data['nvsfac'] = abs(data.nvsfc)
    data['hvsfac'] = abs(data.hvsfc)

    return data


def add_two_stimulus_means(data):
    # needs cleanup
    data[ujoin(difference_list[0], 'mean')] = data[["pfix_control", "pfix_neutral"]].mean(axis=1)
    data[ujoin(difference_list[1], 'mean')] = data[["pfix_control", "pfix_happy"]].mean(axis=1)
    data[ujoin(difference_list[2], 'mean')] = data[["pfix_control", "pfix_fearful"]].mean(axis=1)
    data[ujoin(difference_list[3], 'mean')] = data[["pfix_neutral", "pfix_happy"]].mean(axis=1)
    data[ujoin(difference_list[4], 'mean')] = data[["pfix_neutral", "pfix_fearful"]].mean(axis=1)
    data[ujoin(difference_list[5], 'mean')] = data[["pfix_happy", "pfix_fearful"]].mean(axis=1)

    data['cnc_avg'] = data[["weighted_control", "weighted_neutral"]].mean(axis=1)
    data['chc_avg'] = data[["weighted_control", "weighted_happy"]].mean(axis=1)
    data['cfc_avg'] = data[["weighted_control", "weighted_fearful"]].mean(axis=1)
    data['nhc_avg'] = data[["weighted_neutral", "weighted_happy"]].mean(axis=1)
    data['nfc_avg'] = data[["weighted_neutral", "weighted_fearful"]].mean(axis=1)
    data['hfc_avg'] = data[["weighted_happy", "weighted_fearful"]].mean(axis=1)

    return data


def weight_with_success_percentage(data):
    for i in range(1,5):
        data[ujoin('weighted', ids[i])] = data[ujoin('pfix', ids[i]) * data[ujoin('success_percentage', ids[i])]]
    return data


def pfix_differences(data):
    data = weight_with_success_percentage(data)
    data = add_differences(data)
    data = add_two_stimulus_means(data)
    return data


def worst_performance(group_trials, data):
    subject_list = data.subject.tolist()
    subject_worst_performance = []

    for i,s in enumerate(subject_list):
        sub = group_trials[group_trials.subject == subject_list[i]]
        subject_worst_performance.append(np.percentile(np.sort(sub.srt.dropna().tolist()), 90))

    data['subject_worst_performace'] = subject_worst_performance
    data['subject_worst_performace_zscore'] = st.zscore(subject_worst_performance)

    return data


def data_transform(data, difference):
    # CLEAN UP NEEDED, this is terrible
    if difference == "cvsna":
        diff_label = data.cn_avg
        stimuli_mean = "Control vs Neutral"
    elif difference == "cvsha":
        stimuli_mean = data.ch_avg
        diff_label = "Control vs Happy"
    elif difference == "cvsfa":
        stimuli_mean = data.cf_avg
        diff_label = "Control vs Fearful"
    elif difference == "nvsha":
        stimuli_mean = data.nh_avg
        diff_label = "Neutral vs Happy"
    elif difference == "nvsfa":
        stimuli_mean = data.nf_avg
        diff_label = "Neutral vs Fearful"
    elif difference == "hvsfa":
        stimuli_mean = data.hf_avg
        diff_label = "Neutral vs Happy"
    elif difference == "cvsnac":
        stimuli_mean = data.cnc_avg
        diff_label = "Weighted Control vs Neutral"
    elif difference == "cvshac":
        stimuli_mean = data.chc_avg
        diff_label = "Weighted Control vs Happy"
    elif difference == "cvsfac":
        stimuli_mean = data.cfc_avg
        diff_label = "Weighted Control vs Fearful"
    elif difference == "nvshac":
        stimuli_mean = data.nhc_avg
        diff_label = "Weighted Neutral vs Happy"
    elif difference == "nvsfac":
        stimuli_mean = data.nfc_avg
        diff_label = "Weighted Neutral vs Fearful"
    elif difference == "hvsfac":
        stimuli_mean = data.hfc_avg
        diff_label = "Weighted Happy vs Fearful"

    w_stat, p_value = st.shapiro(data[difference])

    print(diff_label + " Shapiro-Wilk test of normality, W: " + str(w_stat) + ", p-value: " + str(p_value))
    if p_value < 0.05:
        print("The distribution is non-normal, proceeding to square root transform")
        transformed_result = square_transform(data, difference)
    else:
        print("The distribution is (approx.) normal, proceeding to curvefitting")
        transformed_result = curvefitting(data, stimuli_mean, diff_label, difference)

    return transformed_result


def curvefitting(data, stimuli_mean, diff_label, difference):
    # clean up if possible --> probably not necessary?
    x = []
    y = []
    points = []
    idx = data.index.tolist()

    # get avg of control and neutral trials and their difference per subject
    for i in idx:
        lst = []
        lst.append(stimuli_mean[i])
        lst.append(data[difference][i])
        lst.append(data.subject[i])
        points.append(lst)
    points.sort(key=lambda x: x[0])
    orig_order = []
    for lst in points:
        orig_order.append(lst[2])

    # fit regression curve
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    fit, res, _, _, _ = np.polyfit(x, y, 2, full=True)
    fit_fn = np.poly1d(fit)  # regression formula
    print(diff_label)
    print(fit_fn)

    #plot
    #plt.plot(x, y, 'yo', x, fit_fn(x), 'k--')
    #plt.title(diff_label + " " + str(fit_fn))
    #plt.xlabel("Mean " + diff_label)
    #plt.ylabel("Abs. difference " + diff_label)
    #plt.savefig(ujoin(stimuli, "reg.png"))
    #plt.show()

    #quadratic model
    qmodel = QuadraticModel()
    pars = qmodel.guess(y, x=x)
    result = qmodel.fit(y, pars, x=x)
    dely = result.eval_uncertainty(sigma=3) #uncertainty of the model at each point

    #plt.plot(x, y, 'bo')
    #plt.title("lmfit regression")
    #plt.plot(x, result.init_fit, 'k--')
    #plt.plot(x, result.best_fit, 'r-')
    #plt.show()

    #?
    new_even = y - result.init_fit

    #plt.plot(new_even, 'bo')
    #plt.title(diff_label + " residuals")
    #plt.plot([0, 38], [0, 0], 'k--', lw=1)
    #plt.savefig(ujoin(stimuli, "res.png"))
    #plt.show()

    new_order = []
    for i, v in enumerate(new_even):
        help_list = []
        help_list.append(v)
        help_list.append(orig_order[i])
        new_order.append(help_list)
    new_order.sort(key=lambda x: x[1])

    new_resids = [item[0] for item in new_order]
    data[ujoin(difference, "resid")] = new_resids
    print(result.fit_report(min_correl=0.25))

    return new_resids
    #add desc statistics


def square_transform(data, difference):
    transformed = np.sqrt(data[difference].tolist())
    print("Normality after transform: " + str(st.shapiro(transformed)))
    data[ujoin(difference, "sqrt")] = transformed
    return transformed


def percentiles(data, difference):
    zeroth_percentile = np.percentile(data[difference], 2.5)
    first_percentile = np.percentile(data[difference], 16)
    second_percentile = np.percentile(data[difference], 50)
    third_percentile = np.percentile(data[difference], 84)
    fourth_percentile = np.percentile(data[difference], 97.5)
    fifth_percentile = np.percentile(data[difference], 100)

    return [zeroth_percentile, first_percentile, second_percentile, third_percentile, fourth_percentile, fifth_percentile]


def percentile_plots(data):
    differences = ["nvsha", "nvsfa", "hvsfa"]
    for i,r in data.iterrows():
        for d in differences:
            percentiles = percentiles(data, d)
            print(percentiles)
            x = range(1)
            plt.plot(r[d],x, "o")
            plt.title(r.subject + d)
            plt.axvspan(-1, perc[0], color='red', alpha=0.3)
            plt.axvspan(perc[0], perc[1], color='red', alpha=0.1)
            plt.axvspan(perc[3], perc[4], color='red', alpha=0.1)
            plt.plot([perc[2], perc[2]],[-0.5,0.5], 'k--')
            plt.axvspan(perc[4], perc[5] + 0.02, color='red', alpha=0.3)
            plt.axis([-0.02,perc[5] + 0.02,-0.5,0.5])
            plt.savefig(ujoin(r.subject, d, "percentiles.png"))
            plt.show()


def transform_all(data):
    differences = ["cvsna","cvsha","cvsfa","nvsha","nvsfa","hvsfa","cvsnac","cvshac","cvsfac","nvshac","nvsfac","hvsfac"]
    for d in differences:
        arvot = dataTransform(data, d)
        data[ujoin(d, "fin_z")] = st.zscore(arvot)


def save_file(data, filename):
    data.to_csv(filename, encoding='utf-8')


def print_profiles(data):
    for i, r in data.iterrows():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([r.pfix_c, r.pfix_n, r.pfix_h, r.pfix_f], 'o-')
        plt.plot([-1, 7], [r.pfix, r.pfix], 'k--', lw=1, label="mean " + str(round(r.pfix, 2)))
        plt.title(r.subject + " pfix", fontsize=14)
        plt.axis([-1, 4, 0, 1])
        plt.xticks(range(4), ["Control", "Neutral", "Happy", "Fearful", np.nan])
        plt.xlabel("Stimulus type", fontsize=10)
        plt.ylabel("Pfix", fontsize=10)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(10)
        plt.savefig(ujoin((r.subject, "pfix.png")))
        plt.legend()
        plt.show()

        meandiff1 = np.mean([r.cvsna, r.cvsha, r.cvsfa, r.nvsha, r.nvsfa, r.hvsfa])
        meandiff2 = np.mean([r.cvsnac, r.cvshac, r.cvsfac, r.nvshac, r.nvsfac, r.hvsfac])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([r.cvsna, r.cvsha, r.cvsfa, r.nvsha, r.nvsfa, r.hvsfa], 'bo-')
        ax.plot([r.cvsnac, r.cvshac, r.cvsfac, r.nvshac, r.nvsfac, r.hvsfac], 'ro-')
        plt.plot([-1, 7], [meandiff1, meandiff1], 'b--', lw=1, label="mean " + str(round(meandiff1, 2)))
        plt.plot([-1, 7], [meandiff2, meandiff2], 'r--', lw=1, label="mean " + str(round(meandiff2, 2)))
        plt.axis([-1, 6, 0, 1])
        plt.title(r.subject + " (absolute) stimulus differences", fontsize=14)
        plt.xticks(range(6),
                   ["  Control -\nNeutral", "  Control -\nHappy", "  Control -\nFearful", "  Neutral -\nHappy",
                    "  Neutral -\nFearful", "  Happy -\nFearful", np.nan], fontsize=8)
        plt.xlabel("Stimuli", fontsize=10)
        plt.ylabel("Difference", fontsize=10)
        for label in ax.get_yticklabels():
            label.set_fontsize(10)
        plt.savefig(ujoin((r.subject, "difference.png")))
        plt.legend()
        plt.show()

        x = range(11)
        fig = plt.figure(figsize=(6, 14))
        ax = fig.add_subplot(111)
        ax.plot([r.hvsfa_fin_z, np.mean([r.nvsfa_fin_z, r.nvsha_fin_z]), r.nvsfa_fin_z, r.nvsha_fin_z, np.mean([r.cvsfa_fin_z, r.cvsha_fin_z, r.cvsna_fin_z]), r.cvsfa_fin_z, r.cvsha_fin_z, r.cvsna_fin_z, r.wp90z,
                 r.best_z, r.srtZ], x, marker="o", color="k", linewidth=0, label="orig")
        plt.plot([r.hvsfac_fin_z, np.mean([r.nvsfac_fin_z, r.nvshac_fin_z]), r.nvsfac_fin_z, r.nvshac_fin_z, np.mean([r.cvsfac_fin_z, r.cvshac_fin_z, r.cvsnac_fin_z]),r.cvsfac_fin_z, r.cvshac_fin_z, r.cvsnac_fin_z, np.nan, np.nan, np.nan], x, marker="x",
                 color="g", linewidth=0, label="weighted pfix")
        plt.axvspan(1.96, 3.7, color='red', alpha=0.3)
        plt.axvspan(-1.96, -3.7, color='red', alpha=0.3)
        plt.axvspan(1, 3.7, color='red', alpha=0.1)
        plt.axvspan(-1, -3.7, color='red', alpha=0.1)
        plt.axis([-3.7, 3.7, -1, 11])
        plt.plot([0, 0], [-1, 11], 'k--', lw=1)
        plt.plot([-3.7, 3.7], [0.5, 0.5], 'k-', lw=1)
        plt.plot([-3.7, 3.7], [3.5, 3.5], 'k-', lw=1)
        plt.plot([-3.7, 3.7], [4.5, 4.5], 'k-', lw=1, alpha=0.3)
        plt.axhspan(4.5,3.5, color="m", alpha=0.1)
        plt.plot([-3.7, 3.7], [7.5, 7.5], 'k-', lw=1)
        plt.axhspan(0.5, 1.5, color="m", alpha=0.1)
        plt.plot([-3.7, 3.7], [1.5, 1.5], 'k-', lw=1, alpha=0.3)
        plt.ylabel("Index")
        plt.xlabel("z-score")
        plt.title(r.subject + " eye tracking profile")
        plt.yticks(range(11), ['H vs F', "EMOTION", "N vs F", "N vs H", "FACE", "C vs F", "C vs H", "C vs N", "SRT 90p", "SRT best", "SRT med"])
        plt.savefig(ujoin(r.subject, "profile.png"))
        plt.show()


def remove_subjects(data, subject_list):
    for s in subject_list:
        data = data[data.subject != s]

    return data


def get_stimulustypes(sub):
    control = sub[(sub.condition == 'control.bmp')]
    neutral = sub[(sub.condition == 'neutral.bmp')]
    happy = sub[(sub.condition == 'happy.bmp')]
    fearful = sub[(sub.condition == 'fearful.bmp')]

    return [control, neutral, happy, fearful]


def ujoin(string1, string2):
    return '_'.join(string1, string2)

def sjoin(string1, string2):
    return '/'.join(string1, string2)

#def best_times():
#uudetBest = []
#for i,r in data.iterrows():
#    p50 = r.SRTmed
#    sub = sA[sA.subject == r.subject]
#    subSRT = [value for value in sub.srtAll if not math.isnan(value)]
#    bestSRT = [value for value in subSRT if value <= p50]
#    uudetBest.append(np.mean(bestSRT))
#data['best_srt'] = uudetBest
#data["best_z"] = st.zscore(data.best_srt)
#print(np.mean(data.best_srt))