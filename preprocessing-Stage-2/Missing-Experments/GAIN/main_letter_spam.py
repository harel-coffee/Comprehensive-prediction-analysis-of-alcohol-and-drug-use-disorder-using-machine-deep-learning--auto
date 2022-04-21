'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd 
from data_loader import data_loader
from gain import gain
from utils import rmse_loss


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  print(data_m.shape,'*******')

  dict =  {  'DAST': miss_data_x[:,0],
            'SEX': miss_data_x[:,1],
            'HISPANIC': miss_data_x[:,2],
            'RACE': miss_data_x[:,3],
            'VET': miss_data_x[:,4],
            
            'ACTIVE': miss_data_x[:,5],
            'DEPLOY': miss_data_x[:,6],
            'AUDIT': miss_data_x[:,7],


            'COSCREEN': miss_data_x[:,8],    
            'BI': miss_data_x[:,9],
            'BT': miss_data_x[:,10],
            'RT': miss_data_x[:,11],
            
            'ANYALC': miss_data_x[:,12],
            'BINGEDAYS': miss_data_x[:,13],
            'DRUGDAYS': miss_data_x[:,14],

            'ALCDRUGS': miss_data_x[:,15],
            'DAYSCOCAINE': miss_data_x[:,16],
            'MARYJDAYS': miss_data_x[:,17],



            'METHDAYS': miss_data_x[:,18],
            'INJECT': miss_data_x[:,19],
            'AGE': miss_data_x[:,20],
            'TOBMONTH': miss_data_x[:,21],         

         }
  dict1 =  {'DAST':ori_data_x[:,0],
            'SEX': ori_data_x[:,1],
            'HISPANIC': ori_data_x[:,2],
            'RACE': ori_data_x[:,3],
            'VET': ori_data_x[:,4],

            'ACTIVE': ori_data_x[:,5],
            'DEPLOY': ori_data_x[:,6],
            'AUDIT': ori_data_x[:,7],
            'COSCREEN': ori_data_x[:,8],
            'BI': ori_data_x[:,9],
            'BT': ori_data_x[:,10],
            'RT': ori_data_x[:,11],

            'ANYALC': ori_data_x[:,12],            
            'BINGEDAYS': ori_data_x[:,13],
            'DRUGDAYS': ori_data_x[:,14],

            'ALCDRUGS': ori_data_x[:,15],
            'DAYSCOCAINE': ori_data_x[:,16],
            'MARYJDAYS': ori_data_x[:,17],


            'METHDAYS': ori_data_x[:,18],

            'INJECT': ori_data_x[:,19],
            'AGE': ori_data_x[:,20],
            'TOBMONTH': ori_data_x[:,21],         

         }

 


  Columns=['DAST', 'SEX', 'HISPANIC',  'RACE', 'VET','ACTIVE', 'DEPLOY', 'AUDIT','COSCREEN', 
       'BI', 'BT', 'RT',  'ANYALC', 'BINGEDAYS','DRUGDAYS',
       'ALCDRUGS', 'DAYSCOCAINE', 'MARYJDAYS', 'METHDAYS',  'INJECT', 'AGE', 'TOBMONTH']       
  dp = pd.DataFrame(dict,columns=Columns)
  dp1 = pd.DataFrame(dict1,columns=Columns)
  dp1.to_csv("data/original_data.csv")
  dp.to_csv("data/missing_data.csv")
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters)
  
  # Report the RMSE performance
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['Xdata'],
      default='Xdata',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
