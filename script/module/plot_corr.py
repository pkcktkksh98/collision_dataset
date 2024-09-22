import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(data_corr):
    fig1,ax = plt.subplots(figsize=(10,10))
    sns.heatmap(abs(data_corr.corr(method = "pearson")),annot=True,ax=ax)
    fig2 = sns.pairplot(data_corr,vars = [ 'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT',
       'INJURIES', 'SERIOUSINJURIES', 'JUNCTIONTYPE', 'SDOT_COLCODE',
       'UNDERINFL', 'LIGHTCOND', 'PEDROWNOTGRNT', 'ST_COLCODE',
       'HITPARKEDCAR'], hue = 'SEVERITYCODE')
    return fig1,fig2