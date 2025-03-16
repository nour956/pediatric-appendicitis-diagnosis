import pytest
import pandas as pd
import numpy as np
from data_processing import optimize_memory

def get_memory_usage(df):
    return df.memory_usage(deep=True).sum() / (1024**2)  

@pytest.fixture
def sample_dataframe():
    data = {
        'int_column': np.random.randint(1, 100, size=1000000),
        'float_column': np.random.random(size=1000000),
        'string_column': ['text'] * 1000000
    }
    return pd.DataFrame(data)

def test_memory_optimization(sample_dataframe):
    
    before_mem = get_memory_usage(sample_dataframe)
    print(f"Before optimization: {before_mem:.2f} MB")
    
    optimized_df = optimize_memory(sample_dataframe)
    
    after_mem = get_memory_usage(optimized_df)
    print(f"After optimization: {after_mem:.2f} MB")
    
    assert after_mem < before_mem, "Memory usage did not decrease after optimization."
