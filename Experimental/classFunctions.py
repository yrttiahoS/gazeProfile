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

class subject:
    number = ''
    srt_median = 0
    srt_best = 0
    srt_90p = 0

    def print_subject(self):
        desc = "%s is a %s %s worth $%.2f." % (self.number, self.sert_median, self.kind, self.value)
        return desc


def plot_style():
    plt.style.use(['default', 'seaborn-ticks'])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 7
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size


def add_all_files(dir, subdir):
    df = pd.DataFrame()
    filepath = '/'.join(dir, subdir)
    for file in [f for f in listdir(filepath) if isfile(join(filepath, file))]:
        df = df.append(pd.read_csv('/'.join(filepath, file)), ignore_index=True)
    df[df < 0] = np.nan # nan is better than -1
    return df


def get_controls(datafolder, task_type):
    data = add_all_files(datafolder, task_type)

    data['subject'] = df.filename.str.partition("_")[0]

    if task_type == 'SRT':
        group_trials, group_subjects = srt_analysis(data)
    elif task_type == 'Face':   
        group_trials, group_subjects = face_analysis(data)

    return group_subjects, group_trials


def analysis_start(df):
    group_trials = pd.DataFrame() # replace with something else?
    group_subjects = pd.DataFrame()
    return group_trials, group_subjects, subjects


def srt_analysis(df):
    group_trials, group_subjects
    subjects = np.unique(df.subject)

    for index, subject in enumerate(subjects):
        
        subject_data = pd.DataFrame()
        sub = df[(df.subject == subject)]

        subject_data['srt_trials'] = sub.combination # in the data preprocessing, combination = reaction time
        subject_data[sdf.srt_trials == 1000] = np.nan # reaction times of 1000 ms are not valid (no gaze shift) --> remove
        subject_data['subject'] = [subject] * len(subject_data.index)

        group_trials = group_trials.append(sdf, ignore_index=True)  # Adds subject's trial-by-trial data to group df

        # Combined data values
        sdf['missing'] = sdf.srt_all.isnull().sum()  # Counts amount of missing values
        sdf['srt_med'] = sdf.srt_all.median()  # Subject median of all trials

        sdf.drop(['srt_all', 'trial'], axis=1, inplace=True, errors='ignore')  # Drop trial-by-trial data
        subject_out = pd.DataFrame(sdf.iloc[0]).T  # Gets only first row, since all remaining values are the same
        group_subjects = group_subjects.append(subject_out, ignore_index=True)  # Adds row to subject data frame

        group_subjects['srt_z'] = st.zscore(group_subjects.srt_med)  # Calculates z-scores of subject means from all trials

        return group_trials, group_subjects