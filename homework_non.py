import numpy as np
import pandas as pd
import statsmodels.api as sm

lt_beta1=[]
lt_beta2=[]
for i in range(100):
    u=np.random.normal(0,1,100)
    v=np.random.normal(0,1,100)
    w=np.random.normal(0,1,100)
#y=0.4+x1*0.7+x2*0.7+w
    x1=3*u
    x2=3*v
    x1=x1.reshape((1,100))
    x2=x2.reshape((1,100))
    concatenate=np.concatenate([x1,x2],axis=0)
    x=sm.add_constant(concatenate.T)
    beta=[0.4,0.7,0.7]
    y=np.dot(x,beta)+w
    results = sm.OLS(y,x).fit()
    print(results.params)
    lt_beta1.append(results.params[1])
    lt_beta2.append(results.params[2])
arr_beta1=np.array(lt_beta1)
arr_beta2=np.array(lt_beta2)
s_squared_beta1=np.sum((arr_beta1-np.mean(arr_beta1))**2)/(len(lt_beta1)-1)
s_squared_beta2=np.sum((arr_beta2-np.mean(arr_beta2))**2)/(len(lt_beta2)-1)
print(s_squared_beta1,s_squared_beta2)
writer=pd.ExcelWriter(r'C:\Users\Administrator\Desktop\data_1.xlsx')
data=pd.DataFrame({'beta1':arr_beta1,'beta2':arr_beta2})
data.to_excel(writer,'Sheet1')
writer.save()
