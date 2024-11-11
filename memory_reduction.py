def shrink_memory(df):

  """
    It tries to reduce the memory usage of the dataframe
    Parameters: Dataframe
    Return: Dataframe
    """
  start_mem_usg = df.memory_usage().sum() / 1024**3
  print("Memory usage of orignal data is :", round(start_mem_usg , 2)," GB")
  for col in df.columns:
      if df[col].dtypes in ["int64", "int32", "int16"]:

          cmin = df[col].min()
          cmax = df[col].max()

          if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
              df[col] = df[col].astype(np.int8)

          elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
              df[col] = df[col].astype(np.int16)

          elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
              df[col] = df[col].astype(np.int32)

      if df[col].dtypes in ["float64", "float32"]:

          cmin = df[col].min()
          cmax = df[col].max()

          if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
              df[col] = df[col].astype(np.float16)
          # elif cmin > np.finfo(np.float8).min and cmax < np.finfo(np.float8).max:
          #     df[col] = df[col].astype(np.float8)
          elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
              df[col] = df[col].astype(np.float32)


  print("")
  print("Memory after reduction without loss in precision")
  mem_usg = df.memory_usage().sum() / 1024**3
  print("Memory usage is: ",round(mem_usg , 2)," GB")
  print("This is ",100* round(mem_usg/start_mem_usg , 2),"% of the initial size")

  return df
