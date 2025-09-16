import io 
import pandas as pd

def get_df_info(df: pd.DataFrame) -> str:
    
    buffer = io.StringIO()  
    df.info(buf=buffer) 
    return buffer.getvalue()