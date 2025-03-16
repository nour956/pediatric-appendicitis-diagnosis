import pandas as pd
import numpy as np
def optimize_memory(df):
    start_mem = df.memory_usage(deep=True).sum() / (1024**2)  
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
   
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object':
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
    
    end_mem = df.memory_usage(deep=True).sum() / (1024**2)  # Memory usage in MB
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    
    return df