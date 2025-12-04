import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt

if __name__ == '__main__':
    data = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv')

    print(data.head())

    group_30 = data[data['version'] == 'gate_30']
    group_40 = data[data['version'] == 'gate_40']

    metrics = ['retention_1', 'retention_7']

    for metric in metrics:        
        n_30 = len(group_30)
        successes_30 = group_30[metric].sum()
        
        n_40 = len(group_40)
        successes_40 = group_40[metric].sum()
        
        with pm.Model() as model:
            p_30 = pm.Beta('p_30', alpha=1, beta=1)
            p_40 = pm.Beta('p_40', alpha=1, beta=1)
            
            obs_30 = pm.Binomial('obs_30', n=n_30, p=p_30, observed=successes_30)
            obs_40 = pm.Binomial('obs_40', n=n_40, p=p_40, observed=successes_40)
            
            diff = pm.Deterministic('diff', p_40 - p_30)
            
            trace = pm.sample(draws=2000, tune=1000, chains=2, progressbar=True)
            
        summary = az.summary(trace, var_names=['p_30', 'p_40', 'diff'])
        print(summary)
        
        posterior_diff = trace.posterior['diff'].values.flatten()
        prob_40_better = (posterior_diff > 0).mean()
        
        print(f"Probability that gate_40 has higher {metric}: {prob_40_better:.2%}")
        print(f"Mean effect size: {posterior_diff.mean():.4%}")