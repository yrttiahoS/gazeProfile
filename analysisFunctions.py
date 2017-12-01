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

#### DO NOT CHANGE THESE FUNCTIONS, ONLY THE ANALYSIS FILE ####

def plot_style():
    plt.style.use(['default', 'seaborn-ticks'])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 7
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size


def get_control_data(data_dir):
    control_srt_subjects, control_srt_trials = get_control_data(data_dir, 'SRT')
    control_face_subjects, control_face_trials = get_control_data(data_dir, 'Face')

    control_data = pd.merge(control_srt_subjects, control_face_subjects, on='subject', how="right")
    #control_data_trials = pd.merge(control_srt_trials, control_face_trials, on='subject', how="right") # Currently not needed

    return control_data #, control_data_trials


def add_all_files(data_dir, subdir):
    df = pd.DataFrame()
    filepath = '/'.join(dir, subdir)
    for file in [f for f in listdir(filepath) if isfile(join(filepath, file))]:
        df = df.append(pd.read_csv('/'.join(filepath, file)), ignore_index=True)
    df[df < 0] = np.nan # nan is better than -1
    return df


def get_control_data(datafolder, task_type):
    data = add_all_files(datafolder, task_type)

    data['subject'] = df.filename.str.partition("_")[0]

    if task_type == 'SRT':
        group_trials, group_subjects = srt_analysis(data)
    elif task_type == 'Face':   
        group_trials, group_subjects = face_analysis(data)

    return group_subjects, group_trials


def analysis_files(df):
    group_trials = pd.DataFrame()
    group_subjects = pd.DataFrame()
    subject_list = np.unique(df.subject)

    return group_trials, group_subjects, subject_list


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
            subject_data[miss_list[i]] = subject_data[shown_list[i]] / sdf[success_list[i]]

        group_trials = group_trials.append(subject_data.loc[:,['subject', 'stimulus', 'trial_all', 'trial_control', 'trial_neutral', 'trial_happy', 'trial_fearful']], ignore_index=True)
        subject_data.drop(['stimulus', 'trial_all', 'trial_control', 'trial_neutral', 'trial_happy', 'trial_fearful'], axis=1, inplace=True, errors='ignore') # this is a bit copy-paste-y, please fix
        group_subjects = group_subjects.append(subject_data.head(1), ignore_index=True)

    return group_subject_out, group_all_out


def get_stimuli_lists():
    ids = ['all', 'control', 'neutral', 'happy', 'fearful']
    filters, pfix, success, shown, success_percentage, miss = ([] for x in range(6))

    for i in ids:
        filters.append("_".join(['trial', i]))
        pfix.append("_".join(['pfix', i]))
        success.append("_".join(['success', i]))
        shown.append("_".join(['shown', i]))
        success_percentage.append("_".join(['success_percentage', i]))
        miss.append("_".join(['miss', i]))

    return stimuli, filters, pfix, success, shown, success_percentage, miss


def get_srt_std(group_trials, datas): #rename?
    subject_std = []
    sub_list = datas.subject.tolist()
    for s in sub_list:
        sub = group_trials[group_trials.subject == s]
        subject_std.append(np.std(sub.srt.dropna()))
    datas['srt_std'] = subject_std
    square_transformed = np.sqrt(datas.srt_std)

    print("Shapiro-Wilk test of normality for non-transformed data: " + str(st.shapiro(datas.srt_std)))
    print("Shapiro-Wilk test of normality for square root transformed data: " + str(st.shapiro(square_transformed))

    square_zscore = st.zscore(square_transformed)
    datas['sqrt_std'] = square_transformed
    datas['sqrt_std_z'] = square_zscore
    
    return datas

# Unfinished but currently not needed
#def getCI(group_trials, datas):
    #return "this is unfinished"

def add_basic_differences(datas, difference_list):
    datas[difference_list[0]] = datas.pfix_control - datas.pfix_neutral
    datas[difference_list[1]] = datas.pfix_control - datas.pfix_happy
    datas[difference_list[2]] = datas.pfix_control - datas.pfix_fearful
    datas[difference_list[3]] = datas.pfix_neutral - datas.pfix_happy
    datas[difference_list[4]] = datas.pfix_neutral - datas.pfix_fearful
    datas[difference_list[5]] = datas.pfix_happy - datas.pfix_fearful

    for i in range(6):
        datas['_'.join(difference_list[i], 'absolute')] = abs(datas.difference_list[i])

    return datas


def add_two_stimulus_means(datas, difference_list):
    datas['_'.join(difference_list[0], 'mean')] = datas[["pfix_c", "pfix_n"]].mean(axis=1)
    datas['_'.join(difference_list[1], 'mean')] = datas[["pfix_c", "pfix_h"]].mean(axis=1)
    datas['_'.join(difference_list[2], 'mean')] = datas[["pfix_c", "pfix_f"]].mean(axis=1)
    datas['_'.join(difference_list[3], 'mean')] = datas[["pfix_n", "pfix_h"]].mean(axis=1)
    datas['_'.join(difference_list[4], 'mean')] = datas[["pfix_n", "pfix_f"]].mean(axis=1)
    datas['_'.join(difference_list[5], 'mean')] = datas[["pfix_h", "pfix_f"]].mean(axis=1)


def pfix_differences(datas):
    difference_list = ['control_neutral', 'control_happy', 'control_fearful', 'neutral_happy', 'neutral_fearful', 'happy_fearful']

    datas = add_basic_differences(datas, difference_list)
    datas = add_two_stimulus_means()

    #weighting with success perventage
    datas['comb_c'] = datas.pfix_c * datas.c_succp
    datas['comb_n'] = datas.pfix_n * datas.n_succp
    datas['comb_h'] = datas.pfix_h * datas.h_succp
    datas['comb_f'] = datas.pfix_f * datas.f_succp

    #difference of weighted values
    datas['cvsnc'] = datas.comb_c - datas.comb_n
    datas['cvshc'] = datas.comb_c - datas.comb_h
    datas['cvsfc'] = datas.comb_c - datas.comb_f
    datas['nvshc'] = datas.comb_n - datas.comb_h
    datas['nvsfc'] = datas.comb_n - datas.comb_f
    datas['hvsfc'] = datas.comb_h - datas.comb_f

    #absolute differences
    datas['cvsnac'] = abs(datas.cvsnc)
    datas['cvshac'] = abs(datas.cvshc)
    datas['cvsfac'] = abs(datas.cvsfc)
    datas['nvshac'] = abs(datas.nvshc)
    datas['nvsfac'] = abs(datas.nvsfc)
    datas['hvsfac'] = abs(datas.hvsfc)

    #mean of weighted stimuli
    datas['cnc_avg'] = datas[["comb_c", "comb_n"]].mean(axis=1)
    datas['chc_avg'] = datas[["comb_c", "comb_h"]].mean(axis=1)
    datas['cfc_avg'] = datas[["comb_c", "comb_f"]].mean(axis=1)
    datas['nhc_avg'] = datas[["comb_n", "comb_h"]].mean(axis=1)
    datas['nfc_avg'] = datas[["comb_n", "comb_f"]].mean(axis=1)
    datas['hfc_avg'] = datas[["comb_h", "comb_f"]].mean(axis=1)


def wpr(sA, datas):
    sub_list = datas.subject.tolist()
    sub_wpr = []

    for i,s in enumerate(sub_list):
        sub = sA[sA.subject == sub_list[i]]
        sub_wpr.append(np.percentile(np.sort(sub.srtAll.dropna().tolist()), 90))

    datas['wp90'] = sub_wpr
    datas['wp90z'] = st.zscore(sub_wpr)


def dataTransform(datas, diff):
    if diff == "cvsna":
        ka = datas.cn_avg
        seli = "Control vs Neutral"
    elif diff == "cvsha":
        ka = datas.ch_avg
        seli = "Control vs Happy"
    elif diff == "cvsfa":
        ka = datas.cf_avg
        seli = "Control vs Fearful"
    elif diff == "nvsha":
        ka = datas.nh_avg
        seli = "Neutral vs Happy"
    elif diff == "nvsfa":
        ka = datas.nf_avg
        seli = "Neutral vs Fearful"
    elif diff == "hvsfa":
        ka = datas.hf_avg
        seli = "Neutral vs Happy"
    elif diff == "cvsnac":
        ka = datas.cnc_avg
        seli = "Weighted Control vs Neutral"
    elif diff == "cvshac":
        ka = datas.chc_avg
        seli = "Weighted Control vs Happy"
    elif diff == "cvsfac":
        ka = datas.cfc_avg
        seli = "Weighted Control vs Fearful"
    elif diff == "nvshac":
        ka = datas.nhc_avg
        seli = "Weighted Neutral vs Happy"
    elif diff == "nvsfac":
        ka = datas.nfc_avg
        seli = "Weighted Neutral vs Fearful"
    elif diff == "hvsfac":
        ka = datas.hfc_avg
        seli = "Weighted Happy vs Fearful"

    w, p = st.shapiro(datas[diff])
    #sns.distplot(datas[diff])
    #plt.show()

    print(seli + " Shapiro-Wilk test of normality, W: " + str(w) + ", p-value: " + str(p))
    if p < 0.05:
        print("The distribution is non-normal, proceeding to square root transform")
        tulos = sqrtTrans(datas, diff)
    else:
        print("The distribution is (approx.) normal, proceeding to curvefitting")
        tulos = curvefitting(datas, ka, seli, diff)

    return tulos


def curvefitting(datas, ka, seli, diff):
    x = []
    y = []
    points = []
    idx = datas.index.tolist()

    # get avg of control and neutral trials and their difference per subject
    for i in idx:
        lst = []
        lst.append(ka[i])
        lst.append(datas[diff][i])
        lst.append(datas.subject[i])
        points.append(lst)
    points.sort(key=lambda x: x[0])
    jarjestys = []
    for lst in points:
        jarjestys.append(lst[2])

    # fit regression curve
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    fit, res, _, _, _ = np.polyfit(x, y, 2, full=True)
    fit_fn = np.poly1d(fit)  # regression formula
    print(seli)
    print(fit_fn)

    #plot
    #plt.plot(x, y, 'yo', x, fit_fn(x), 'k--')
    #plt.title(seli + " " + str(fit_fn))
    #plt.xlabel("Mean " + seli)
    #plt.ylabel("Abs. difference " + seli)
    #plt.savefig("".join([stimuli, "_reg.png"]))
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

    tasotus = y - result.init_fit

    #plt.plot(tasotus, 'bo')
    #plt.title(seli + " residuals")
    #plt.plot([0, 38], [0, 0], 'k--', lw=1)
    #plt.savefig("".join([stimuli, "_res.png"]))
    #plt.show()

    uusijarjestys = []
    for i, v in enumerate(tasotus):
        plorp = []
        plorp.append(v)
        plorp.append(jarjestys[i])
        uusijarjestys.append(plorp)
    uusijarjestys.sort(key=lambda x: x[1])

    uudet = [item[0] for item in uusijarjestys]
    datas["_".join([diff, "resid"])] = uudet
    print(result.fit_report(min_correl=0.25))

    return uudet
    #add desc statistics


def sqrtTrans(datas, diff):
    uusi = np.sqrt(datas[diff].tolist())
    print("Normality after transform: " + str(st.shapiro(uusi)))
    datas["_".join([diff, "sqrt"])] = uusi
    return uusi


def percentiles(datas, diff):
    nolla = np.percentile(datas[diff], 2.5)
    eka = np.percentile(datas[diff], 16)
    toka = np.percentile(datas[diff], 50)
    kolmas = np.percentile(datas[diff], 84)
    neljas = np.percentile(datas[diff], 97.5)
    viides = np.percentile(datas[diff], 100)

    return nolla, eka, toka, kolmas, neljas, viides


def percPlots(datas):
    diffs = ["nvsha", "nvsfa", "hvsfa"]
    for i,r in datas.iterrows():
        for d in diffs:
            perc = percentiles(datas, d)
            print(perc)
            x = range(1)
            plt.plot(r[d],x, "o")
            plt.title(r.subject + d)
            plt.axvspan(-1, perc[0], color='red', alpha=0.3)
            plt.axvspan(perc[0], perc[1], color='red', alpha=0.1)
            plt.axvspan(perc[3], perc[4], color='red', alpha=0.1)
            plt.plot([perc[2], perc[2]],[-0.5,0.5], 'k--')
            plt.axvspan(perc[4], perc[5] + 0.02, color='red', alpha=0.3)
            plt.axis([-0.02,perc[5] + 0.02,-0.5,0.5])
            plt.savefig("".join([r.subject, d, "_percentiles.png"]))
            plt.show()


def transformAll(datas):
    differences = ["cvsna","cvsha","cvsfa","nvsha","nvsfa","hvsfa","cvsnac","cvshac","cvsfac","nvshac","nvsfac","hvsfac"]
    for d in differences:
        arvot = dataTransform(datas, d)
        datas["_".join([d, "fin_z"])] = st.zscore(arvot)


def saveFile(df, filename):
    df.to_csv(filename, encoding='utf-8')


def printAllProfiles(datas):
    for i, r in datas.iterrows():
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
        plt.savefig("".join((r.subject, "_pfix.png")))
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
        plt.savefig("".join((r.subject, "_diff.png")))
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
        #plt.legend()
        plt.yticks(range(11), ['H vs F', "EMOTION", "N vs F", "N vs H", "FACE", "C vs F", "C vs H", "C vs N", "SRT 90p", "SRT best", "SRT med"])
        plt.savefig("".join((r.subject, "_profile.png")))
        plt.show()


def removeSubjects(datas, subjectList):
    for s in subjectList:
        datas = datas[datas.subject != s]

    return datas


def get_stimulustypes(sub):
    control = sub[(sub.condition == 'control.bmp')]
    neutral = sub[(sub.condition == 'neutral.bmp')]
    happy = sub[(sub.condition == 'happy.bmp')]
    fearful = sub[(sub.condition == 'fearful.bmp')]

    return [control, neutral, happy, fearful]
