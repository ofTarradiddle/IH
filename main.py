
import pandas as pd 
import numpy as np 
import argparse 
import pickle
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import argparse 

# helper lambda functions
poly     = lambda s,p: s ** p 
rsquared = lambda x,y: np.corrcoef(x, y)[0,1]**2

def get_data(file = "data.xlsx"):
    """Read in tables from excel 

    Args:
        file (str, optional): _description_. Defaults to "data.xlsx".

    Returns:
        _type_: Merged dataset 
    """
    P, B = pd.read_excel(file, sheet_name="performance_evaluations", header=1),pd.read_excel("data.xlsx", sheet_name="background")

    P.head() # teacher_id years -> 
    B.head() # teacher_id training_year board_cert_year

    Plong = pd.melt(P, id_vars='teacher_id', value_vars=[x for x in range(1971,2003)])

    # # one observation per teacher_id
    # B.groupby(['teacher_id'])['training_year'].count().sort_values()
    # B.groupby(['teacher_id'])['board_cert_year'].count().sort_values()

    # MERGE
    return B.merge(Plong, how = "left", on = "teacher_id", validate = "1:m")

def transform(D):
    
    D = D.copy() # dont modify original 

    for c in ['variable','training_year','board_cert_year']:
        if ~isinstance(D[c], float):
            D.loc[:,c] = D.loc[:,c].astype(float)

    # There are NaNs if year is less than cert year 
    D = D.assign(years_since_train = lambda x: x.variable - x.training_year, 
                    # years_since_train2 = lambda x: x.years_since_train * x.years_since_train, 
                    years_since_cert = lambda x: x.variable - x.board_cert_year, 
                    # years_since_cert2 = lambda x: x.years_since_cert ^ 2, 

                    before1990 = lambda x: x.variable < 1990, 
                    cert_is_train_year = lambda x: x.board_cert_year == x.training_year,
                    is_cert = lambda x: x.years_since_cert >= 0, 
                    on_second_try = lambda x: x.board_cert_year == x.training_year + 1 ,

                    renewal_5yr = lambda x: ((x.variable  - x.board_cert_year) % 5 == 0) & (x.variable > 1990),
                    last_year_renewed = lambda x:  (x.variable > 1990) & ((x.variable  - x.board_cert_year) % 5 == 1),
                    second_renewal = lambda x:  (x.variable > 1990) & ((x.variable  - x.board_cert_year) % 10 == 1),
                    third_renewal = lambda x:  (x.variable > 1990) & ((x.variable  - x.board_cert_year) % 15 == 1),
                    fourth_renewal = lambda x:  (x.variable > 1990) & ((x.variable  - x.board_cert_year) % 20 == 1),
                    cert_mod = lambda x: x.years_since_cert % 5 == 1,
                    hasValue = lambda x: ~np.isnan(x.value)
                    )
    D.groupby(['years_since_train'])['value'].count()
    teachers = D.dropna().groupby(['teacher_id'])['teacher_id'].count().sort_values().where(lambda x : x > 0).dropna().index

    D['years_since_train'] = D['years_since_train'].astype(float)
    D['years_since_cert'] = D['years_since_cert'].astype(float)
    D['last_year_performance'] = D.sort_values(['variable'],ascending=True).groupby(['teacher_id'])['value'].shift(1)
    D = D[D['hasValue']] # Only keep observations where there is a scores, this mostly accounts for years before training for each teacher 

    # dummies_df = pd.get_dummies(D.teacher_id)
    bool_cols = ['cert_mod',"second_renewal","third_renewal","fourth_renewal",'before1990','cert_is_train_year','is_cert','on_second_try','renewal_5yr','last_year_renewed',]
    D.loc[:,bool_cols] =  D.get(bool_cols).applymap(int)


    D = D.query("teacher_id in @teachers")
    return D 

def show_me(df, teachers):
    # Initialize the figure style
    plt.style.use('seaborn-darkgrid')
    
    # create a color palette
    palette = plt.get_cmap('Set1')
    
    # multiple line plot
    for i,t in enumerate(teachers):

        r = D.loc[lambda x: x.teacher_id == t]
        series, dates, bcy, ty = r['value'], r['variable'], r['board_cert_year'].unique(), r['training_year'].unique()
        plt.plot(dates, series, marker='', color=palette(i), linewidth=1.9, alpha=0.9, label=id)

    """    
    # general title
    plt.suptitle("Teacher Performance By Year", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
    
    # Axis titles
    plt.text(0.5, 0.02, 'Time', ha='center', va='center')
    plt.text(0.06, 0.5, 'Preformance', ha='center', va='center', rotation='vertical')

    # Show the graph
    plt.show()"""



    return D

def plot_averages(D):
    mean_df = D.groupby(['years_since_cert','before1990'])['value'].agg({'mean','count'}).reset_index().sort_values(['before1990','years_since_cert'], ascending = [False, True])
    for c in ['mean']:
        for i,n in enumerate([0,1]):
            temp = mean_df.loc[lambda x: x.before1990 == n].copy()
            x,y = temp.years_since_cert, temp[c]
            if c == 'mean':
                plt.plot(x,y)
            else:
                plt.bar(x,y/100)

def retiree_data(D):

        retire_year = D.groupby(['teacher_id'])['years_since_cert'].max().reset_index()
        
        retired = D.merge(retire_year, on = ['teacher_id','years_since_cert'], how = 'inner',validate = "1:1")
        
        print("How many teachers retire each year, except last year ")
        retired.groupby(['variable'])['variable'].count().iloc[:-1] # remove last year 

        retired_count = retired.query("variable == 1989.0").groupby(['years_since_train'])['value'].count()
        count = D.query("variable == 1989.0").groupby(['years_since_train'])['value'].count()
        
        print("Proportionate retirees, effects disproportionately folks further into their career as percent, but smaller total, but likely just an effect of smaller pool ")
        retired_count / count # 

def show_stats(D):
        return   (D.query("before1990 == True")
                    .query("training_year == board_cert_year")
                    .groupby(['years_since_train'])['value']
                    .agg({'mean','std','count'}))

def group_means(D):
        stats =  D.groupby(['before1990','cert_is_train_year'])['value'].aggregate({'count','median','mean','std','max','min'})
        stats['count'][0][0]/stats['count'][0].sum() # after 1990 
        stats['count'][1][0]/stats['count'][1].sum() # before 1990 

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--model', help='Use OLS or Random Effects', required=True, default = "LR")
parser.add_argument('-v','--verbose', help='Show plots ', required=True, default=True)
args = vars(parser.parse_args())

if __name__ == "__main__":

    D = get_data()

    D = transform(D)

    # LR 
    if args.model == "LR":

        X = D.get(['before1990','cert_is_train_year','renewal_5yr','last_year_performance','years_since_cert']).fillna(0)

        # Linear Regression
        y = D.get(['value']).fillna(0)
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        print(est2.summary())

        import statsmodels.stats.api as sms
        from statsmodels.graphics.regressionplots import plot_leverage_resid2

        name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
        test = sms.jarque_bera(est2.resid)

        # Jarque Bera tests whether skew and kurtosis match a normal distribution 
        # null hypothesis is a joint H of skew being 0 and (excess) kurtosis being 3 (0). 
        # Chi**2 fo 0 indicates unable to reject the null
        list(zip(name,test)) 

        # # Leverage plot 
        # fig, ax = plt.subplots(figsize=(8, 6))
        # fig = plot_leverage_resid2(est2, ax=ax)

        # Condition number 
        np.linalg.cond(est2.model.exog)

        # constant Variance 
        name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
        test = sms.het_breuschpagan(est2.resid, est2.model.exog)
        list(zip(name, test))

        # Linearity
        name = ["t value", "p value"]
        test = sms.linear_harvey_collier(est2)
        list(zip(name, test))
    else:
        # Random Effects model
        Dm = D.copy()
        Dm['last_year_performance'] = Dm.sort_values(['variable'],ascending=True).groupby(['teacher_id'])['value'].shift(1)
        Dm = Dm.dropna()

        md = smf.mixedlm("value ~ 1 + years_since_train  + before1990 + cert_is_train_year", data=Dm, groups=Dm["teacher_id"]) # , re_formula="~years_since_train")
        mdf = md.fit()
        print(mdf.summary())

        fitted = md.predict(mdf.fe_params, exog=Dm.get(['years_since_train','before1990','cert_is_train_year','value']))
        true = Dm['value']

        print(f"Rquared is {rsquared(fitted, true):.4f}")

    # most of variation is explained by teacher, 
    # controller for teacher_id via a fixed effects model we see that only cert is train year has a significant effect

    #means drift slighly higher, likely a factor of small group sizes for people late into career who are good at their jobs 
    if args.verbose:
        plot_averages(D)
    #retirees 
    retiree_data(D)

    show_stats(D)

    group_means(D)




