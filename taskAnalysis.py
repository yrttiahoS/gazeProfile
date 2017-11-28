from analysisFunctions import *
import seaborn as sns

#make the plots pretty
plotStyle()

#read control group files for both tasks
controlSRTS, controlSRTA = readFile(controlData, 'SRT') #filenames of preprocessed _CONTROL_ SRT-data
controlFaceS, controlFaceA = readFile(['disengagement_tbt_face_korj_vanha.csv', 'disengagement_tbt_face_korj_uusi.csv'], 'Face') #filenames of preprocessed _CONTROL_ Face-data

#combine SRT and Face task datas into one
datas = pd.merge(sS, fS, on='subject', how="right")
datas = datas[(datas.subject != 'TV33') & (datas.subject != 'TV59') & (datas.missing.notnull())]
print("Verrokkien n = " + str(len(datas)))

#getStd(sA, datas) #get SRT standard deviations
#getCI(sA, datas) #get SRT confidence intervals (KESKEN!!)

print(sA)
srtmedian = np.mean(datas.SRTmed)
sns.distplot(datas.SRTmed.dropna())
plt.ylim(0,0.006)
plt.plot([srtmedian, srtmedian], [0,0.008], 'k--',label="mean " + str(srtmedian) + " ms")
plt.xlabel("Saccadic Reaction Time (ms)")
plt.legend()
plt.show()

#90. percentile for subject SRTs
wpr(sA, datas)

uudetBest = []
for i,r in datas.iterrows():
    p50 = r.SRTmed
    sub = sA[sA.subject == r.subject]
    subSRT = [value for value in sub.srtAll if not math.isnan(value)]
    bestSRT = [value for value in subSRT if value <= p50]
    uudetBest.append(np.mean(bestSRT))
datas['best_srt'] = uudetBest
datas["best_z"] = st.zscore(datas.best_srt)
print(np.mean(datas.best_srt))

#One way ANOVA for
fVal, pVal = st.f_oneway(datas.pfix_c, datas.pfix_n, datas.pfix_h, datas.pfix_f)
print("Pfix ANOVA F: " + str(fVal) + ", p-value: " + str(pVal))

#Deltas for each stimulus pair
pfixDeltas(datas)

#Test distribution normality, if normal --> fit quadratic regression curve to delta x the mean of the two pfix means
transformAll(datas)
#print("persentiilit")
#percPlots(datas)

plt.plot([np.mean(datas.pfix_c), np.mean(datas.pfix_n), np.mean(datas.pfix_h), np.mean(datas.pfix_f)], 'bo-')
plt.plot([np.mean(datas.comb_c), np.mean(datas.comb_n), np.mean(datas.comb_h), np.mean(datas.comb_f)], 'ro-')
plt.axis([-0.5,3.5,0,1])
plt.xticks([0,1,2,3], ["Kontrolli", "Neutraali", "Iloinen", "Pelokas"])
plt.xlabel("Kasvojen ilme")
plt.ylabel("Katseluosuus")
plt.show()

#saveFile(datas, "verrokit.csv")

#printAllProfiles(datas)

#TOIBILAS
#tSS, tSA = readToibFile(['epi9_2_SRT.csv'], 'SRT') #filenames of preprocessed SRT-data
#tFS, tFA = readToibFile(['dtbt_toibilas_face_25s.csv', 'dtbt_toibilas_srt_28s.csv', 'dtbt_toibilas_srt_33s.csv'], 'Face')

#print(tSS)
#print(tSA)
#tdatas = pd.merge(tSS, tFS, on='subject', how="right")
#tdatas = tdatas[(tdatas.subject != 'Toib36') & (tdatas.subject != 'Toib25') & (tdatas.index != 2)]
#wpr(tSA, tdatas)
#pfixDeltas(tdatas)
#transformToib(tdatas)
