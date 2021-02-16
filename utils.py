# Data Manipulation & Summarisation
import numpy as np 
from numpy import sqrt, abs, round
import pandas as pd

# debugging imports
from IPython import embed
import pdb

# Data Visualisation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
color = sns.color_palette()
# %matplotlib inline

# Regex for working with text features
import re

# Datetime for working with datetime features
from datetime import timedelta, date

# Modules required for statistical tests
from scipy.stats import norm
from scipy.stats import t as t_dist
from scipy.stats import chi2_contingency

# -----------------------------------------------------------------------------------------------------------------------
# The following utility functions below were scrapped from presenters of Analytics Vidhya. Repurposed for this assement.
# ------------------------------------------------------------------------------------------------------------------------

# Custom function for easy visualisation of Categorical Variables - super convenient function to look through categorical features one at a time.
def EDA_category(data, var_group):

    '''
    Univariate_Analysis_categorical
    takes a group of variables (category) and plot/print all the value_counts and barplot.
    '''
    # setting figure_size
    size = len(var_group)
    
    fig = plt.figure(figsize = (7*size,5), dpi = 100)
    fig.canvas.draw()
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=100000, rect=None)

    # for every variable
    for i, col_name in enumerate(var_group):
        
        # Plotting the variable with every information
        plt.subplot(1,size,i+1)
        
        # from IPython import embed; embed()
        ax = sns.countplot(x=col_name, data=data, orient = 'h')        
        
        plt.xlabel('{}'.format(col_name), fontsize = 12)
    
        ncount = data.shape[0] - pd.isnull(data[col_name]).sum()
        
        # Make twin axis
        ax2=ax.twinx()

        # Switch so count axis is on right, frequency on left
        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()

        # Also switch the labels over
        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')

        ax2.set_ylabel('Frequency [%]')

        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}'.format(100.*y/ncount), (x.mean(), y), 
                    ha='center', va='bottom') # set the alignment of the text

        # Use a LinearLocator to ensure the correct number of ticks
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))

        # Fix the frequency range to 0-100
        ax2.set_ylim(0,100)
        ax.set_ylim(0,ncount)

        # And use a MultipleLocator to ensure a tick spacing of 10
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

        # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
        ax2.grid(alpha = 0.1)

        plt.show

# custom function for easy and efficient analysis of numerical univariate
def EDA_numeric(data, var_group, log = False):
    
    size = len(var_group)
    plt.figure(figsize = (7*size,3), dpi = 100)
      #looping for each variable
    for j,i in enumerate(var_group):

        # calculating descriptives of variable
        mini = data[i].min()
        maxi = data[i].max()
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
       # calculating points of one standard deviation
        points = mean-st_dev, mean+st_dev

        #Plotting the variable with every information
        plt.subplot(1,size,j+1)
        ax = sns.kdeplot(data[i], shade=True)
        
        if log == True:
            ax.set_xscale('log')
        else:
            pass

        plt.xlabel('{}'.format(i), fontsize = 13)
        plt.ylabel('density')
        plt.title('std_dev = {};range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),
                                                                                                       
                                                                                                       (round(mini,2),round(maxi,2)),
                                                                                                       round(mean,2),
                                                                                                       round(median,2)))
# Bivariate Cont Cat Exploration Function
def Bivariate_cont_cat(data, cont, cat, category):
  #creating 2 samples
    x1 = data[cont][data[cat]==category][:]
    x2 = data[cont][~(data[cat]==category)][:]
    
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.std()
    #calculating p-values
    t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
    z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
    plt.figure(figsize = (15,6), dpi=140)
    #barplot
    plt.subplot(1,2,1)
    sns.barplot([str(category),'not {}'.format(category)], [m1, m2])
    plt.ylabel('mean {}'.format(cont))
    plt.xlabel(cat)
    plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('categorical boxplot')

# Bivariate Exploration Function
def BVA_categorical_plot(data, tar, cat, sort = False):
    temp = data.copy()
    if sort == True:
        df = data[[cat, tar]].groupby([cat])[tar] \
                             .count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False)
        temp = temp.merge(df, on = cat, how = 'left')
        data = temp.sort_values(by = 'count', ascending = False)
        data.drop(cat,axis = 1, inplace=True)
        data[cat] = data['count'].rank(ascending = False)
    else:
        pass
    data = data[[cat,tar]][:]

  #forming a crosstab
    table = pd.crosstab(data[tar],data[cat],)
    f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])

  #performing chi2 test
    chi, p, dof, expected = chi2_contingency(f_obs)
    
    if p<0.05:
        sig = True
    else:
        sig = False

  #plotting grouped plot
    sns.countplot(x=cat, hue=tar, data=data)
    plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

  #plotting percent stacked bar plot
  #sns.catplot(ax, kind='stacked')
    ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
    ax1.plot(kind='bar', stacked='True',title=str(ax1))
    int_level = data[cat].value_counts()

def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval

def TwoSampT(X1, X2, sd1, sd2, n1, n2):
    ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
    t = (X1 - X2)/ovr_sd
    df = n1+n2-2
    pval = 2*(1 - t_dist.cdf(abs(t),df))
    return pval

