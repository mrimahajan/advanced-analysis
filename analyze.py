import pandas as pd
import numpy as np
from scipy.stats import skew,kurtosis
from scipy.special import cbrt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chisquare
from matplotlib import pyplot as plt
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as f

def numerical_categorical_division(df):
    	numerical,categorical = [],[]
    	for col in df.columns:
    	    if 'int' in str(df[col].dtypes) or 'float' in str(df[col].dtypes):
    	        numerical.append(col)
    	    else:
    	        categorical.append(col)
    	return numerical,categorical

def edd(data,dv=None,regression=True,percentile=[.01,.05,.1,.5,.9,.95,.99],cv=[2,3]):
    numerical,categorical = numerical_categorical_division(data)
    df_desc = data.describe().transpose()
    df_desc['Var'] = df_desc.index
    df_desc.reset_index(inplace=True)
    df_desc.drop('count',axis=1,inplace=True)
    df_desc['skewness'] = df_desc['Var'].apply(lambda x: skew(np.array(data.loc[data[x].notnull(),x])))
    df_desc['kurtosis'] = df_desc['Var'].apply(lambda x: kurtosis(np.array(data.loc[data[x].notnull(),x]),fisher=False))
    for pct in percentile:
        df_desc['p'+str(int(pct*100))] = df_desc['Var'].apply(lambda x: data[x].quantile(pct))
    
    for dev in cv:
        df_desc['mean-'+str(int(dev))+'sigma'] = df_desc['mean'] - dev*df_desc['std']
        df_desc['mean+'+str(int(dev))+'sigma'] = df_desc['mean'] + dev*df_desc['std']
    
    df_desc['type']='numeric'

    df_categorical = pd.DataFrame()
    df_categorical['Var']=np.array(categorical)
    df_categorical['type']='categorical'
    for col in [c for c in df_desc.columns if c not in ['Var','type']]:
        df_categorical[col]=np.nan
    for col in categorical:
        df_var = data[col].value_counts(ascending=True,dropna=False).cumsum()/data.shape[0]
        df_cat = pd.DataFrame(df_var)
        df_cat.reset_index(inplace=True)
        df_cat.columns = ['categories','cum_pct']
        df_categorical.loc[df_categorical['Var']==col,'min'] = list(df_cat['categories'])[0]
        df_categorical.loc[df_categorical['Var']==col,'max'] = list(df_cat['categories'])[-1]
        for pct in percentile:
            df_categorical.loc[df_categorical['Var']==col,'p'+str(int(pct*100))] = list(df_cat.loc[df_cat['cum_pct']>= pct,'categories'])[0]

  
        del df_var
        del df_cat

    df_categorical = df_categorical[df_desc.columns]
    edd = pd.concat([df_desc,df_categorical])
    del df_desc
    del df_categorical
    edd['count'] = edd['Var'].apply(lambda x: data[data[x].notnull()].shape[0])
    edd['nmiss'] = data.shape[0]-edd['count']
    edd['missing_rate'] = np.array(edd['nmiss']).astype('float')/data.shape[0] * 100
    edd['unique'] = edd['Var'].apply(lambda x: len(data[x].value_counts().index.tolist()))
    col_list = ['Var','type','count','nmiss','missing_rate','unique','std','skewness','kurtosis','mean','min'] + \
    ['mean-'+str(int(dev))+'sigma' for dev in cv] + ['p'+str(int(pct*100)) for pct in percentile] + \
    ['mean+'+str(int(dev))+'sigma' for dev in cv] + ['max']

    edd = edd[col_list]

    if dv:
        edd['correlation/p_value'] = np.nan
        if regression==True:
            corr_matrix = data.corr()
            for col in numerical:
                edd.loc[edd['Var']==col,'correlation/p_value'] = corr_matrix.loc[col,dv]
            for col in categorical:
                mod = ols(dv+' ~ '+ col,data=data).fit()
                aov_table = sm.stats.anova_lm(mod,type=2)
                edd.loc[edd['Var']==col,'correlation/p_value'] = aov_table.loc[col,'PR(>F)']

        else:
            for col in numerical:
                mod = ols(col+' ~ '+ dv,data=data).fit()
                aov_table = sm.stats.anova_lm(mod,type=2)
                edd.loc[edd['Var']==col,'correlation/p_value'] = aov_table.loc[dv,'PR(>F)']

            for col in categorical:
                f =pd.crosstab(data[dv],data[col],dropna=False)
                edd.loc[edd['Var']==col,'correlation/p_value'] = chisquare(np.reshape(np.array(f),np.product(f.shape))).pvalue
    edd.reset_index(inplace=True)
    edd.drop('index',axis=1,inplace=True)
    return edd


##path = folder where you want to save your plots and tables
def graphical_analysis(data,dv,path='',regression=True):
    numerical,categorical = numerical_categorical_division(data)
    if regression:
        for col in numerical:
            if col != dv:
                ax = data.plot(col,dv)
                fig = ax.get_figure()
                fig.savefig(path+col+'.png',dpi=1000)
        for col in categorical:
            if col != dv:
                ax = data.boxplot(dv,by=col)
                fig = ax.get_figure()
                fig.savefig(path+col+'.png',dpi=1000)
    else:
        for col in numerical:
            if col != dv:
                ax = data.boxplot(col,by=dv)
                fig = ax.get_figure()
                fig.savefig(path+col+'.png',dpi=1000)
        for col in categorical:
            if col != dv:
                f =pd.crosstab(data[dv],data[col],dropna=False)
                for cat in f.columns:
                    f[cat] = f[cat].apply(lambda x: x/f[cat].sum()*100)
                ax = plt.subplot(111, frame_on=False) # no visible frame
                ax.xaxis.set_visible(False)  # hide the x axis
                ax.yaxis.set_visible(False)  # hide the y axis

                table(ax, f)  

                plt.savefig(path+col+'.png',dpi=1000)


def trends_generation(data,header,buckets,output='trends.xlsx'):
    numericals = [col for col in list(header.columns) if header.loc[0,col]=='N']
    categoricals = [col for col in list(header.columns) if header.loc[0,col]=='C']
    label_cols = [col for col in list(header.columns) if header.loc[0,col]=='L'][0]

    data.cache()
    data.count()
    data.registerTempTable('data')
    numtr_columns = ['variable','bucket','minima','var_sum','maxima','counts','dep_sum']
    numerical_trends = pd.DataFrame(columns=numtr_columns)

    buckets = buckets
    for col in numericals:
        try:
            print(col)
            print('\n')
            counts = data.filter('{col} is not null and {col} <> 0'.format(col=col)).filter(~f.isnan(col)).count()
            data1 = data.filter('{col} is not null and {col} < 0'.format(col=col)).filter(~f.isnan(col))
            neg_counts = data1.count()
            try:
                neg_buckets = int(np.ceil(buckets*neg_counts/counts))
                print(neg_buckets)
            except Exception as e:
                print(e)
                neg_buckets = 1
            print(neg_buckets)
            if neg_buckets == 0 or np.isnan(neg_buckets):
                neg_buckets = 1
            print(neg_buckets)
            values = sorted(list(data1.select(col).distinct().toPandas().values))
            values = sorted([float(x) for x in values])
            data1.registerTempTable('data1')
            bucket_var = """case
            """
            buck_till = 0
            if neg_counts > 0:
                if np.log10(-values[0]) <= 0:
                    values = sorted([-10**np.ceil(np.log10(-values[0]))/neg_buckets*i for i in range(neg_buckets,0,-1)])
                    for i,val in enumerate(values):
                        bucket_var += """when {col} <= {val} then {i}
                        """.format(col=col,val=val,i=buck_till+i+1)
                    buck_till = buck_till + len(values)
                    bucket_var += """when {col} < 0 then {i}
                    """.format(col=col,i=buck_till+1)
                    buck_till = buck_till + 1
                
                else:
                    buck_row = np.ceil(data1.count()/neg_buckets)
                    df = sqlContext.sql("""select floor(final.rk/{buck_row})+1 as bucket,
                    min(final.{col}) as minima,
                    max(final.{col}) as maxima
                    from
                    (
                    select {col},row_number() over(order by {col}) as rk
                    from data1
                    ) as final
                    group by 1
                    """.format(col=col,buck_row=buck_row)).toPandas().sort_values(by='bucket')

                    bucket2 = 1
                    df['bucket2'] = 0
                    flag = 0
                    for bucket in sorted(list(df['bucket'])):
                        if flag == 0:
                            df.loc[df['bucket']==bucket,'bucket2'] = bucket2
                        try:
                            if float(df.loc[df['bucket']==bucket,'minima']) == float(df.loc[df['bucket']==bucket+1,'minima']) and float(df.loc[df['bucket']==bucket,'maxima']) == float(df.loc[df['bucket']==bucket+1,'maxima']):
                                df.loc[df['bucket']==bucket+1,'bucket2'] = bucket2
                                flag = 1
                            else:
                                bucket2 = bucket2 + 1
                                flag = 0
                        except:
                            pass

                    min_max_lst = []

                    df = df.groupby('bucket2').agg({'minima' : ['min'],'maxima' : ['max']})
                    df.columns = df.columns.droplevel()
                    df.reset_index(inplace=True)
                    df = df.rename({'bucket2':'bucket'},axis=1)

                    for bucket in sorted(list(df['bucket'])):
                        if bucket == 1:
                            minima = float(df.loc[df['bucket']==bucket,'min'])
                            maxima = float(df.loc[df['bucket']==bucket,'max'])
                            min_max_lst.append((minima,maxima))
                            min_max_lst = sorted(min_max_lst,key=lambda x: x[1])
                        else:
                            minima = float(df.loc[df['bucket']==bucket,'min'])
                            maxima = float(df.loc[df['bucket']==bucket,'max'])
                            prev_max = float(min_max_lst[-1][1])
                            if minima == prev_max:
                                if values.index(minima) < len(values) -1:
                                    pos = values.index(prev_max)
                                    minima = float(values[pos+1])
                                    if minima > maxima:
                                        continue
                                    min_max_lst.append((minima,maxima))
                                    min_max_lst = sorted(min_max_lst,key = lambda x: x[1])
                                else:
                                    min_max_lst.append((minima,maxima))
                                    min_max_lst = sorted(min_max_lst,key = lambda x:x[1])

                    for i,x in enumerate(sorted(min_max_lst,key = lambda x: x[1])):
                        minima = x[0]
                        maxima = x[1]
                        bucket_var += """when {col} >= {minima} and {col} <= {maxima} then {i}
                        """.format(col=col,minima=minima,maxima=maxima,i=i+1)
                    bucket_var += """when {col} < 0 then {i}
                    """.format(col=col,i=buck_till+1)
                    buck_till = buck_till + len(min_max_lst)
            bucket_var += """when {col} = 0 then {i}
            """.format(i=buck_till+1,col=col)
            buck_till = buck_till + 1
                

            data1 = data.filter('{col} is not null and {col} > 0'.format(col=col)).filter(~f.isnan(col))
            pos_counts = data1.count()
            try:
                pos_buckets = int(np.ceil(buckets*pos_counts/counts))
            except:
                pos_buckets = 1
            if pos_buckets == 0 or np.isnan(pos_buckets):
                pos_buckets = 1
            values = sorted(list(data1.select(col).distinct().toPandas().values))
            values = list(sorted([float(x) for x in values]))
            data1.registerTempTable('data1')
            if pos_counts > 0:
                if np.log10(values[-1]) <= 0:
                    values = sorted([10**np.ceil(np.log10(values[-1]))/pos_buckets*i for i in range(1,pos_buckets)])
                    for i,val in enumerate(values):
                        bucket_var += """when {col} <= {val} then {i}
                        """.format(col=col,val=val,i=buck_till+i+1)
                    buck_till += len(values)
                    bucket_var += """when {col} < 1 then {j}
                    when {col} = 1 then {i}
                    """.format(col=col,j=buck_till+1,i=buck_till+2)
                else:
                    buck_row = np.ceil(data1.count()/pos_buckets)
                    df = sqlContext.sql("""select floor(final.rk/{buck_row})+1 as bucket,min(final.{col}) as minima,max(final.{col}) as maxima
                    from
                    (
                        select {col},row_number() over(order by {col}) as rk
                        from data1
                    ) as final
                    group by floor(final.rk/{buck_row}) + 1
                    """.format(col=col,buck_row=buck_row)).toPandas().sort_values(by='bucket')

                    bucket2=1
                    df['bucket2'] = 0
                    flag = 0
                    for bucket in sorted(list(df['bucket'])):
                        if flag == 0:
                            df.loc[df['bucket']==bucket,'bucket2'] = bucket2
                        try:
                            if float(df.loc[df['bucket']==bucket,'minima']) == float(df.loc[df['bucket']==bucket+1,'minima']) and float(df.loc[df['bucket']==bucket,'maxima']) == float(df.loc[df['bucket']==bucket+1,'maxima']):
                                df.loc[df['bucket']==bucket+1,'bucket2'] = bucket2
                                flag = 1
                            else:
                                bucket2 = bucket2 + 1
                                flag = 0
                        except:
                            pass
                    min_max_lst = []

                    df = df.groupby('bucket2').agg({'minima':['min'],'maxima':['max']})
                    df.columns = df.columns.droplevel()
                    df.reset_index(inplace=True)
                    df = df.rename({'bucket2' : 'bucket'},axis=1)

                    for bucket in sorted(list(df['bucket'])):
                        if bucket == 1:
                            minima = float(df.loc[df['bucket']==bucket,'min'])
                            maxima = float(df.loc[df['bucket']==bucket,'max'])
                            min_max_lst.append((minima,maxima))
                            min_max_lst = sorted(min_max_lst,key = lambda x: x[1])
                        else:
                            minima = float(df.loc[df['bucket']==bucket,'min'])
                            maxima = float(df.loc[df['bucket']==bucket,'max'])
                            prev_max = float(min_max_lst[-1][1])
                            if minima == prev_max:
                                if values.index(minima) < len(values) - 1:
                                    pos = values.index(prev_max)
                                    minima = float(values[pos+1])
                                    if minima > maxima:
                                        continue
                                    min_max_lst.append((minima,maxima))
                                    min_max_lst = sorted(min_max_lst,key = lambda x: x[1])
                            else:
                                min_max_lst.append((minima,maxima))
                                min_max_lst = sorted(min_max_lst,key = lambda x: x[1])
                    for i,x in enumerate(sorted(min_max_lst,key=lambda x:x[1])):
                        minima = x[0]
                        maxima = x[1]
                        bucket_var += """when {col} >= {minima} and {col} <= {maxima} then {i}
                        """.format(col=col,minima=minima,maxima=maxima,i=buck_till+i+1)
            bucket_var += "end"
                
            query = """select {bucket_var} as bucket,
            min({col}) as minima,sum({col}) as var_sum,max({col}) as maxima,count(*) as counts,sum({label_col}) as lifts
            from data
            group by 1
            """.format(col=col,bucket_var=bucket_var,label_col=label_col)

            print(query)
            df = sqlContext.sql(query).toPandas().sort_values(by='minima')
            df['bucket'] = [i+1 for i in range(df.shape[0])]
            df['variable'] = col
            df = df[numtr_columns]
            numerical_trends = numerical_trends.append(df)
            numerical_trends.to_excel(output,sheet_name='numerical_trends',index=False)
        except Exception as e:
            print(e)
            # query = """select {bucket_var} as bucket,
            # min({col}) as minima,sum({col}) as var_sum,max({col}) as maxima,count(*) as counts,sum({label_col}) as lifts
            # from data
            # group by 1
            # """.format(col=col,bucket_var=bucket_var,label_col=label_col)

            # print(query)

            # df = sqlContext.sql(query).toPandas().sort_values(by='minima')
            # df['bucket'] = [i+1 for i in range(df.shape[0])]
            # df['variable'] = col
            # df = df[numtr_columns]
            # numerical_trends = numerical_trends.append(df)
    numerical_trends['var_avg'] = numerical_trends['var_sum']/numerical_trends['counts']
    numerical_trends['dep_avg'] = numerical_trends['dep_sum']/numerical_trends['counts']
    numerical_trends.to_excel(output,sheet_name='numerical_trends',index=False)
    cat_cols = ['variable','category','counts','dep_sum']
    categorical_trends = pd.DataFrame(columns=cat_cols)
    for col in categoricals:
        try:
            print(col)
            print('\n')
            query = """select '{col}' as variable , {col} as category,
            count(*) as counts, sum({label_col}) as dep_sum
            from data
            group by 1,2
            """.format(col=col,label_col=label_col)

            print(query)
            df = sqlContext.sql(query).toPandas()
            df = df[cat_cols]
            categorical_trends = categorical_trends.append(df)

        except Exception as e:
            print("Error in categorical col {col}".format(col=col))
            print(e)

    categorical_trends['dep_avg'] = categorical_trends['dep_sum']/categorical_trends['counts']
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    numerical_trends.to_excel(writer,sheet_name='numerical_trends',index=False)
    categorical_trends.to_excel(writer,sheet_name='categorical_trends',index=False)
    writer.save()
    writer.close()

