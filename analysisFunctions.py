import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import lmfit
import math
from lmfit.models import QuadraticModel
from os import listdir

#just so the plots look the same on each run
def plotStyle():
    plt.style.use(['default', 'seaborn-ticks'])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 7
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

def readFile(dataFolder, type):
    # READ THE DATA
    df = pd.DataFrame()
    for filename in filenameList:
        df = df.append(pd.read_csv(filename), ignore_index=True) #add new file to end of dataframe
    df[df < 0] = np.nan  # -1 values to nan

    # SUBJECT AND TRIAL INFORMATION
    sub = df.filename.str.partition("_")[0]
    df['subject'] = sub  # slicing the subject number from the filename (the orig. files need to be named "subNumber_something.gazedata")
    subjects = np.unique(df.subject)  # select unique subject names/numbers

    # OUTPUT
    df_sout = pd.DataFrame()  # all subjects' individual results
    df_all = pd.DataFrame()  # all trials from all subjects

    # INDIVIDUAL SUBJECT ANALYSIS
    for i, s in enumerate(subjects):
        sdf = pd.DataFrame()  # analysis df
        sub = df[(df.subject == s)]  # one subject's results

        if type == 'SRT':
            # fill in basic individual trial values
            sdf['srtAll'] = sub.combination  # all reaction times
            sdf[sdf.srtAll == 1000] = np.nan  # remove values of 1000 ms (no gaze shift)
            sdf['subject'] = [s] * len(sdf.index)  # add subject number to df (a bit clunky but works)

            df_all = df_all.append(sdf, ignore_index=True)  # add trial data

            # combined data values
            sdf['missing'] = sdf.srtAll.isnull().sum()  # amount of missing values
            sdf['SRTmed'] = sdf.srtAll.median()  # subject median of all RTs

            sdf.drop(['srtAll', 'trial'], axis=1, inplace=True, errors='ignore')  # drop trial data
            sout = pd.DataFrame(sdf.iloc[0]).T  # get only first row, since all values are the same
            df_sout = df_sout.append(sout, ignore_index=True)  # add row to subject data frame

            df_sout['srtZ'] = st.zscore(df_sout.SRTmed)  # zscores of subject means of all trials

        if type == 'Face':
            ctrl = sub[(sub.condition == 'control.bmp')]  # all control stimuli trials
            ntrl = sub[(sub.condition == 'neutral.bmp')]  # all neutral
            happ = sub[(sub.condition == 'happy.bmp')]  # all happy
            fear = sub[(sub.condition == 'fearful.bmp')]  # all fearful

            stimus = [sub, ctrl, ntrl, happ, fear]
            filts = ['all', 'ctrl', 'ntrl', 'happ', 'fear']
            pfix_list = ['pfix', 'pfix_c', 'pfix_n', 'pfix_h', 'pfix_f']
            succ = ['succ_lkm_all', 'succ_lkm_c', 'succ_lkm_n', 'succ_lkm_h', 'succ_lkm_f']  # number of technically successful trials
            shown = ['all_shown_lkm', 'c_shown_lkm', 'n_shown_lkm', 'h_shown_lkm', 'f_shown_lkm']

            # add all necessary values to dataframe with one loop
            for i, f in enumerate(stimus):
                sdf[filts[i]] = stimus[i].csaccpindex  # all pfix values, including NaN-values by stimulus type
                sdf[pfix_list[i]] = sdf[filts[i]].dropna().mean()  # mean of non-NaN values by stimulus type
                sdf[succ[i]] = stimus[i][stimus[i]['technical error'] == 0]['technical error'].count()  # count of technically successful trials
                sdf[shown[i]] = stimus[i]['technical error'].count()

            # percentages of technically successful trials per stimulus
            ids = ['all', 'c', 'n', 'h', 'f']
            for i,f in enumerate(ids):
                sdf["_".join([f, "succp"])] = sdf["_".join(["succ_lkm",f])] / sdf["_".join([f,"shown_lkm"])]
                sdf["_".join([f, "miss"])] = sdf["_".join([f, "shown_lkm"])] / sdf["_".join(["succ_lkm", f])]

            sdf['stimulus'] = df.condition.str[:-4]
            sdf['subject'] = [s] * len(sdf.index)

            aout = pd.DataFrame(sdf[['subject', 'all', 'stimulus', 'ctrl', 'ntrl', 'happ', 'fear']])  # select only the individual values
            df_all = df_all.append(aout, ignore_index=True)
            sdf.drop(['all', 'ctrl', 'ntrl', 'happ', 'fear', 'stimulus'], axis=1, inplace=True, errors='ignore')  # drop individual trial information
            sout = pd.DataFrame(sdf.iloc[0]).T  # select only first row, since all values are the same
            df_sout = df_sout.append(sout, ignore_index=True)  # append the full df

    return df_sout, df_all

#KORJAA!!!
def readToibFile(filenameList, type):
    if type == 'SRT':
        dfs = pd.DataFrame()
        for filename in filenameList:
            dfs = dfs.append(pd.read_csv(filename), ignore_index=True)  # add new file to end of dataframe
        dfs[dfs < 0] = np.nan  # -1 values to nan

        # SUBJECT AND TRIAL INFORMATION
        dfs['subject'] = dfs.filename.str[:6]  # slicing the subject number from the filename
        dfs['experiment'] = dfs.filename.str[10:11]  # experiment number
        dfs = dfs[(dfs.subject != 'Toib36') & (dfs.subject != 'Toib25')]
        subjects = np.unique(dfs.subject)  # select unique subjects
        exps = [1, 2, 3]

        # OUTPUT
        dfs_sout = pd.DataFrame()  # all subjects' individual results
        dfs_all = pd.DataFrame()  # all trials from all subjects

        # INDIVIDUAL SUBJECT ANALYSIS
        for i, s in enumerate(subjects):
            sdfs = pd.DataFrame()  # analysis df
            sub = dfs[(dfs.subject == s)]  # one subject's results

            # fill in basic individual trial values
            sdfs['srtAll'] = sub.combination  # all reaction times
            sdfs[sdfs.srtAll == 1000] = np.nan  # remove values of 1000ms
            sdfs['subject'] = [s] * len(sdfs.index)  # add subject number to df
            sdfs['trial'] = range(36)

            dfs_all = dfs_all.append(sdfs, ignore_index=True)  # add trial data

            # combined data values
            sdfs['missing'] = sdfs.srtAll.isnull().sum()  # amount of missing values
            sdfs['SRTmed'] = sdfs.srtAll.median()  # subject mean of all RTs

            sdfs.drop(['srtAll', 'trial'], axis=1, inplace=True,
                      errors='ignore')  # drop trial data (if SRTbest used, drop that too)
            sout = pd.DataFrame(sdfs.iloc[0]).T  # get only first row, since all values are the same
            dfs_sout = dfs_sout.append(sout, ignore_index=True)  # add row to subject data frame

        return dfs_sout, dfs_all

    if type == 'Face':
        dff1 = pd.read_csv('dtbt_toibilas_face_25s.csv')
        dff2 = pd.read_csv('dtbt_toibilas_face_28s.csv')
        dff3 = pd.read_csv('dtbt_toibilas_face_33s.csv')
        dff = dff1.append(dff2, ignore_index=True)
        dff = dff.append(dff3, ignore_index=True)
        dff[dff < 0] = np.nan  # -1 values to nan

        # SUBJECT AND TRIAL INFORMATION
        dff['subject'] = dff.filename.str[:6]  # slicing the subject number from the filename
        dff['experiment'] = dff.filename.str[10:11]  # experiment number
        subjects = np.unique(dff.subject).tolist()  # get unique subjects as list
        subjects.pop(0)

        # OUTPUT DATAFRAMES
        dff_sout = pd.DataFrame()  # mean data for each subject
        dff_all = pd.DataFrame()  # all individual values for distribution analysis

        # INDIVIDUAL SUBJECT ANALYSIS
        for i, s in enumerate(subjects):
            sdff = pd.DataFrame()  # create new data frame to hold results

            # logical indexing (slicing data by stimulus)
            sub = dff[(dff.subject == s)]  # all data of one subject
            ctrl = sub[(sub.condition == 'control.bmp')]  # all control stimuli trials
            ntrl = sub[(sub.condition == 'neutral.bmp')]  # all neutral
            happ = sub[(sub.condition == 'happy.bmp')]  # all happy
            fear = sub[(sub.condition == 'fearful.bmp')]  # all fearful

            # list all necessary values for easier looping
            stimus = [sub, ctrl, ntrl, happ, fear]
            filts = ['all', 'ctrl', 'ntrl', 'happ', 'fear']
            pfix_list = ['pfix', 'pfix_c', 'pfix_n', 'pfix_h', 'pfix_f']
            pfix_slist = ['pfix_shift', 'pfix_c_shift', 'pfix_n_shift', 'pfix_h_shift', 'pfix_f_shift']
            obl = ['obl_all', 'obl_c', 'obl_n', 'obl_h',
               'obl_f']  # number of trials where the child didn't shift their gaze ("obligatory looking")
            succ = ['succ_lkm', 'succ_lkm_c', 'succ_lkm_n', 'succ_lkm_h',
                'succ_lkm_f']  # number of technically successful trials
            shown = ['all_shown_lkm', 'c_shown_lkm', 'n_shown_lkm', 'h_shown_lkm', 'f_shown_lkm']
            rts = ['rtime', 'rtime_c', 'rtime_n', 'rtime_h', 'rtime_f']  # reaction times

            # add all necessary values to dataframe with one loop
            for i, f in enumerate(stimus):
                sdff[filts[i]] = stimus[i].csaccpindex  # all pfix values, including NaN-values by stimulus type
                sdff[pfix_list[i]] = sdff[filts[i]].dropna().mean()  # mean of non-NaN values by stimulus type
                try:  # may cause errors if not enough values --> try/except
                    sdff[pfix_slist[i]] = sdff[sdff[filts[i]] < 1][
                    filts[i]].mean()  # mean of pfix values that are less than 1 (meaning there was a gaze shift)
                except:
                    sdff[pfix_slist[i]] = np.nan  # if no shifts, put in NaN
                sdff[obl[i]] = sdff[sdff[filts[i]] == 1][filts[i]].count()  # count of trials where there was no gaze shift
                sdff[succ[i]] = stimus[i][stimus[i]['technical_error'] == 0][
                'technical_error'].count()  # count of technically successful trials
                sdff[shown[i]] = stimus[i]['technical_error'].count()
                sdff[rts[i]] = stimus[i][
                stimus[i].combination < 1000].combination  # reaction times that are faster than 1000ms (=no gaze shift)

            # percentages of technically successful trials per stimulus
            sdff['all_succp'] = sdff.succ_lkm / sdff.all_shown_lkm
            sdff['c_succp'] = sdff.succ_lkm_c / sdff.c_shown_lkm
            sdff['n_succp'] = sdff.succ_lkm_n / sdff.n_shown_lkm
            sdff['h_succp'] = sdff.succ_lkm_h / sdff.h_shown_lkm
            sdff['f_succp'] = sdff.succ_lkm_f / sdff.f_shown_lkm

            sdff['stimulus'] = dff.condition.str[:-4]
            sdff['subject'] = [s] * len(sdff.index)  # subject details

            # separate subject csvs for spss analysis (not necessary?)
            filu = "".join([s, "_faceresults.csv"])
            sdff.to_csv(filu, encoding='utf-8')

            aout = pd.DataFrame(
                sdff[['subject', 'all', 'stimulus', 'ctrl', 'ntrl', 'happ', 'fear']])  # select only the individual values
            dff_all = dff_all.append(aout, ignore_index=True)  # append the
            sdff.drop(['all', 'ctrl', 'ntrl', 'happ', 'fear', 'stimulus'], axis=1, inplace=True,
                  errors='ignore')  # drop individual trial information
            sout = pd.DataFrame(sdff.iloc[0]).T  # select only first row, since all values are the same
            dff_sout = dff_sout.append(sout, ignore_index=True)  # append the full df

        return dff_sout, dff_all

def getStd(sA, datas):
    sstd = []
    sub_list = datas.subject.tolist()
    for s in sub_list:
        sub = sA[sA.subject == s]
        sstd.append(np.std(sub.srtAll.dropna()))
    datas['srt_std'] = sstd #count srt std for each participant

    #sns.distplot(datas.srt_std)
    #plt.title("Subjects' SRT standard deviations")
    #plt.show()
    print("Shapiro-Wilk test of normality: " + str(st.shapiro(datas.srt_std)))

    #sns.distplot(np.sqrt(datas.srt_std))
    #plt.title("Square-transform of subjects' SRT standard deviations")
    #plt.show()
    print("Shapiro-Wilk test of normality: " + str(st.shapiro(np.sqrt(datas.srt_std))))

    sqz = st.zscore(np.sqrt(datas.srt_std))
    datas['sqrt_std'] = np.sqrt(datas.srt_std)
    datas['sqrt_std_z'] = sqz

def getCI(sA, datas):
    return "this is unfinished"

def pfixDeltas(datas):
    #stimulus pifferences
    datas['cvsn'] = datas.pfix_c - datas.pfix_n
    datas['cvsh'] = datas.pfix_c - datas.pfix_h
    datas['cvsf'] = datas.pfix_c - datas.pfix_f
    datas['nvsh'] = datas.pfix_n - datas.pfix_h
    datas['nvsf'] = datas.pfix_n - datas.pfix_f
    datas['hvsf'] = datas.pfix_h - datas.pfix_f

    #absolute differences
    datas['cvsna'] = abs(datas.cvsn)
    datas['cvsha'] = abs(datas.cvsh)
    datas['cvsfa'] = abs(datas.cvsf)
    datas['nvsha'] = abs(datas.nvsh)
    datas['nvsfa'] = abs(datas.nvsf)
    datas['hvsfa'] = abs(datas.hvsf)

    #stimulus mean for fitting the regression curve
    datas['cn_avg'] = datas[["pfix_c", "pfix_n"]].mean(axis=1)
    datas['ch_avg'] = datas[["pfix_c", "pfix_h"]].mean(axis=1)
    datas['cf_avg'] = datas[["pfix_c", "pfix_f"]].mean(axis=1)
    datas['nh_avg'] = datas[["pfix_n", "pfix_h"]].mean(axis=1)
    datas['nf_avg'] = datas[["pfix_n", "pfix_f"]].mean(axis=1)
    datas['hf_avg'] = datas[["pfix_h", "pfix_f"]].mean(axis=1)

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