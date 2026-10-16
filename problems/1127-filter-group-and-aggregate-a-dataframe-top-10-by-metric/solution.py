import pandas as pd

def solution(df):
    df = df[df['status']=='completed']
    result = pd.DataFrame()
    result=df.groupby('region', as_index=False).agg({'amount': 'sum'})
    result = result.sort_values(by='amount', ascending=False)
    result=result[:10]
    return result