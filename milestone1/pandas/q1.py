import pandas as pd 
def load_orders(file_path):

  df = pd.read_csv(file_path)
  df.drop(columns = ['order_date'])
  return df[["order_id"]]


def remove_invalid_order(df):
  df = df.drop_duplicates(subset = "order_id", keep = "first")
  df = df[df['quantity'] > 0]

  return df

def attach_item_details(df, ref_file_path):
  ref_df = pd.read_csv(ref_file_path)
  merge_df = df.merge(ref_df,on="item_id",how="inner")

  return merge_df

def add_order_value_column(df):
  df["order_value"] = (df["quantity"] * df["unit_price"]).round(2)
  return df

def summarize_by_category(df):
  df = df.groupby("category").agg(
    order_value = ("order_value", "sum"),
    quantity = ("quantity", "mean")
  ).reset_index()

#   df = df.groupby("category", as_index = False).agg(
#     order_value = ("order_value", "sum"),
#     quantity = ("quantity", "mean")
#   )

  df['quantity'] = df['quantity'].round(2)

  return df

def format_summary(df):  #rename
    df = df.rename(
        columns = {
            "order_value": "total_revenue",
            "quantity": "avg_quantity",
            "order_id": "order_count",
        }
    )
    df = df.sort_values(by = "category", ascending = True).reset_index(drop = True)

    return df[["category", "total_revenue", "avg_quantity", "order_count"]]

  
