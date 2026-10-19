
def load_visit_records(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns = "visit_date")
    return df

def drop_incomplete_visits(df):
    df = df.drop_duplicates(subset = "visit_id", keep = "first")
    df = df.dropna(subset = ["consultation_fee", "medicine_fee"])
    return df

def attach_Service_details(df, sevice_df):
    merge_df = pd.merge(
        df,
        service_df,
        on = "service_id",
        how = "inner"
    )
    result = merge_df.drop(columns = "service_id")
    return result

def add_total_bill(df):
    df["total_bill"] = (df["consultation_fee"] + df["medicine_fee"]).round(2)
    return df

def summarize_by_city(df):
    df = df.groupby("city", as_index = False).agg(
        avg_bill_amount = {"total_bill", "mean"},
        visit_count = {"visit_id", "count"},
    )
    df["avg_bill_amount"] = df["avg_bill_amount"].round(2)

    df = df.sort_values(by = ["avg_bill_amount", "city"], ascending = [False, True])
    df = df.reset_index(drop = True)

    return df
