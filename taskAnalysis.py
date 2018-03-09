import analysisFunctions as af
for i in dir(af): print( i )


from analysisFunctions import plot_style, get_control_data, remove_subjects, transform_all, percentile_plots, save_file, print_profiles
#worst_performance,pfix_deltas

plot_style()

control_data = get_control_data('controlData')

control_data = remove_subjects(control_data, ['TV33', 'TV59']) # preterms excluded

#getStd(sA, datas)
#getCI(sA, datas)

#pfix_deltas(control_data)

transform_all(control_data)
percentile_plots(control_data)

save_file(control_data, "verrokit.csv")

print_profiles(control_data)

#Individual files
#tSS, tSA = readToibFile(['epi9_2_SRT.csv'], 'SRT') #filenames of preprocessed SRT-data
#tFS, tFA = readToibFile(['dtbt_toibilas_face_25s.csv', 'dtbt_toibilas_srt_28s.csv', 'dtbt_toibilas_srt_33s.csv'], 'Face')

#print(tSS)
#print(tSA)
#tdatas = pd.merge(tSS, tFS, on='subject', how="right")
#tdatas = tdatas[(tdatas.subject != 'Toib36') & (tdatas.subject != 'Toib25') & (tdatas.index != 2)]
#wpr(tSA, tdatas)
#pfixDeltas(tdatas)
#transformToib(tdatas)
