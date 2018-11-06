# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:35:25 2018

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss
from numpy.linalg import inv
#y=beta0+beta1*x1+beta2*x2+w  beta0=0.4,beta1=0.7,beta2=0.7,x1=3v,x2=0.95x1+0.5u
lt_beta0 = []
lt_beta1 = []
lt_beta2 = []
lt_segam_beta0_squared = []
lt_segam_beta1_squared = []
lt_segam_beta2_squared = []
for i in range(1000):
    u=np.random.normal(loc=0,scale=1,size=(100,1))
    v=np.random.normal(loc=0,scale=1,size=(100,1))
    w=np.random.normal(loc=0,scale=1,size=(100,1))
    x1 = 3*v
    x2 = 0.95*x1+0.5*u
    beta0,beta1,beta2 = 0.4,0.7,0.7
    y=beta0+beta1*x1+beta2*x2+w
    beta_vector = np.asarray([0.4,0.7,0.7]) 
    w_vector = w
    y_vector = y
    ones = np.ones_like(x1)
    x_vector = np.array([ones,x1,x2])
    x_vector_flatten = x_vector.flatten()
    x_vector_reshape = x_vector_flatten.reshape((100,3),order='F')
    #y_vector=x_vector_reshape*beta_vector+w_vector
    #beta_vector_estimate=inv(mat)(x_vector_reshape.T)y_vector
    mat = np.dot(x_vector_reshape.T,x_vector_reshape)
    beta_vector_estimate = np.dot(np.dot(inv(mat),x_vector_reshape.T),y_vector)
    y_vector_estimate = np.dot(x_vector_reshape,beta_vector_estimate)
    w_vector_estimate = y_vector-y_vector_estimate
    Var_w_vector = np.dot(w_vector_estimate.T,w_vector_estimate)/(len(y.tolist())-3)
    Var_beta_vector_estimate = Var_w_vector*inv(mat)
    beta0_estimate = beta_vector_estimate.tolist()[0][0]
    beta1_estimate = beta_vector_estimate.tolist()[1][0]
    beta2_estimate = beta_vector_estimate.tolist()[2][0]
    lt_beta0.append(beta0_estimate)
    lt_beta1.append(beta1_estimate)
    lt_beta2.append(beta2_estimate)
    lt_segam_beta0_squared.append('%.8f'%Var_beta_vector_estimate.tolist()[0][0])
    lt_segam_beta1_squared.append('%.8f'%Var_beta_vector_estimate.tolist()[1][1])
    lt_segam_beta2_squared.append('%.8f'%Var_beta_vector_estimate.tolist()[2][2])
    
    
    #y=a0+a1*x1+random_error
    x1_beta1 = np.sum((x1 - np.mean(x1)) * (y - np.mean(y))) / np.sum((x1 - np.mean(x1)) ** 2)
    x1_beta0 = np.mean(y) - x1_beta1 * np.mean(x1)
    y_estimate = x1_beta0 + x1_beta1 * x1
    residual_of_x1_y = y - y_estimate
    R_squared_of_x1_y = np.sum((y_estimate - np.mean(y)) ** 2) / np.sum((y - np.mean(y)) ** 2)
    sigam_squared_estimate_x1_y = np.sum((y - y_estimate) ** 2) / (len(y.tolist()) - 2)
    standard_residual_x1_y = residual_of_x1_y / math.sqrt(sigam_squared_estimate_x1_y)
    s_squared_beta1_x1_y = sigam_squared_estimate_x1_y / np.sum((x1 - np.mean(x1)) ** 2)
    s_squared_beta0_x1_y = sigam_squared_estimate_x1_y * np.sum(x1 * x1) / (len(y.tolist())* np.sum((x1 - np.mean(x1)) ** 2))
    t_value_x1_y_beta1 = x1_beta1 * math.sqrt(np.sum((x1 - np.mean(x1)) ** 2)) / math.sqrt(sigam_squared_estimate_x1_y)
    t_score = ss.t.isf(0.025, df=(len(y.tolist()) - 2))  # t分位值
    p = ss.t.sf(t_value_x1_y_beta1, len(y.tolist()) - 2)  #p值
    r_x1y = np.sum((x1 - np.mean(x1)) * (y - np.mean(y))) / math.sqrt(np.sum((x1 - np.mean(x1)) ** 2) * np.sum((y - np.mean(y)) ** 2))

    
    # y=b0+b1*x2+random_error
    x2_beta1 = np.sum((x1 - np.mean(x1)) * (y - np.mean(y))) / np.sum((x1 - np.mean(x1)) ** 2)
    x2_beta0 = np.mean(y) - x2_beta1 * np.mean(x1)
    y_estimate_x2 = x2_beta0 + x2_beta1 * x2
    residual_of_x2_y = y - y_estimate_x2
    R_squared_of_x2_y = np.sum((y_estimate_x2 - np.mean(y)) ** 2) / np.sum((y - np.mean(y)) ** 2)
    sigam_squared_estimate_x2_y = np.sum((y - y_estimate_x2) ** 2) / (len(y.tolist()) - 2)
    standard_residual_x2_y = residual_of_x2_y / math.sqrt(sigam_squared_estimate_x2_y)
    s_squared_beta1_x2_y = sigam_squared_estimate_x2_y / np.sum((x1 - np.mean(x1)) ** 2)
    s_squared_beta0_x2_y = sigam_squared_estimate_x2_y * np.sum(x1 * x1) / (len(y.tolist()) * np.sum((x1 - np.mean(x1)) ** 2))
    t_value_x2_y_beta1 = x2_beta1 * math.sqrt(np.sum((x1 - np.mean(x1)) ** 2)) / math.sqrt(sigam_squared_estimate_x2_y)
    t_score_x2 = ss.t.isf(0.025, df=(len(y.tolist()) - 2))  # t分位值
    p_x2 = ss.t.sf(t_value_x2_y_beta1, len(y.tolist()) - 2)  # p值
    r_x2y = np.sum((x1 - np.mean(x1)) * (y - np.mean(y))) / math.sqrt(np.sum((x1 - np.mean(x1)) ** 2) * np.sum((y - np.mean(y)) ** 2))
   
    
    fig1 = plt.figure()
    axis1 = fig1.add_subplot(2,2,1)
    axis2 = fig1.add_subplot(2,2,2)
    axis3 = fig1.add_subplot(2,2,3,projection='3d')
    axis4 = fig1.add_subplot(2,2,4,projection='3d')
    
    axis1.scatter(x1,y,marker='o',alpha=0.5,label='Scatter of X1 and Y')
    axis1.plot(x1,y_estimate,color='r',label='Y=a0+a1*X1')
    axis1.set_xlabel('X1')
    axis1.set_ylabel('Y')
    axis1.legend(loc='lower left',bbox_to_anchor=(0.1,0.01,0.4,0.4))
    axis1.text(-6,6.0,'a0=%.8f'%x1_beta0,fontsize=3)
    axis1.text(-6,5.0,'a1=%.8f'%x1_beta1,fontsize=3)
    axis1.text(-6,4.0,'r_X1_and_Y=%.8f'%r_x1y,fontsize=3)
    
    axis2.scatter(x2,y,marker='o',alpha=0.5,label='Scatter of X2 and Y')
    axis2.plot(x2,y_estimate_x2,color='r',label='Y=b0+b1*X2')
    axis2.set_xlabel('X2')
    axis2.set_ylabel('Y')
    axis2.legend(loc='lower left',bbox_to_anchor=(1.0,0.01,0.4,0.4))
    axis2.text(2.0,-6.0,'b0=%.8f'%x2_beta0,fontsize=3)
    axis2.text(2.0,-5.0,'b1=%.8f'%x2_beta1,fontsize=3)
    axis2.text(2.0,-4.0,'r_X2_and_Y=%.8f'%r_x2y,fontsize=3)
    
    axis3.scatter(x1,x2,y)
    axis3.set_xlabel('X1')
    axis3.set_ylabel('X2')
    axis3.set_zlabel('Y')
    
    x1_cook,x2_cook = np.meshgrid(x1,x2)
    y_cook = 0.4+0.7*x1_cook+0.7*x2_cook+w
    axis4.plot_surface(x1_cook,x2_cook,y_cook,cmap=plt.cm.winter)
    axis4.set_xlabel('X1')
    axis4.set_ylabel('X2')
    axis4.set_zlabel('Y')
    
    #plt.savefig(r'C:\Users\Administrator\Desktop\home work for economitrics\work2_fig%d'%(i+10),dpi=400,bbox_inches='tight')
    
#beta(0,1,2)的方差列表
print('list of segam_alpha0_squared is:'+str(lt_segam_beta0_squared),
      'list of segam_alpha1_squared is:'+str(lt_segam_beta1_squared),
      'list of segam_alpha2_squared is:'+str(lt_segam_beta2_squared))
'''save_excel = pd.ExcelWriter(r'C:\Users\Administrator\Desktop\home work for economitrics\data2.xlsx')
data= {'segam_alpha0_squared':lt_segam_beta0_squared,'segam_alpha1_squared':lt_segam_beta1_squared,'segam_alpha2_squared':lt_segam_beta2_squared}
dataframe = pd.DataFrame(data)
dataframe.to_excel(save_excel,'Sheet1')'''

    
#这是10组beta（0,1,2）值作为样本的各个方差   
beta0_vector = np.asarray(lt_beta0)
beta1_vector = np.asarray(lt_beta1)
beta2_vector = np.asarray(lt_beta2)
sample_segam_beta0_squared = np.sum((beta0_vector-np.mean(beta0_vector))**2)/(len(beta0_vector.tolist())-1)
sample_segam_beta1_squared = np.sum((beta1_vector-np.mean(beta1_vector))**2)/(len(beta1_vector.tolist())-1)
sample_segam_beta2_squared = np.sum((beta2_vector-np.mean(beta2_vector))**2)/(len(beta2_vector.tolist())-1)
    
print(sample_segam_beta0_squared,sample_segam_beta1_squared,sample_segam_beta2_squared,len(beta0_vector.tolist()))