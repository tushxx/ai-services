
import pandas as pd


def load_patients(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["last_visit_date"])
    return df[["", "", ""]]

def clean_patient_data(df):
    df = df.drop_duplicates(subset=["patient_id"], keep="first")
    df = df.dropna(subset=["bill_amount"])
    return df


def merge_with_doctors(patients_df, doctors_file_path):
    doctors_df = pd.read_csv(doctors_file_path)

    merge_df =  pd.merge(
        patients_df,
        doctors_df,
        on="doctor_id",
        how="inner"
    )


def aggregate_by_specialty(df):
    return (
        df.groupby("specialty", as_index=False)
        .agg({
            "bill_amount": "mean",
            "patient_id": "count"
        })
    )


def finalize_summary(df):
    df = df.rename(
        columns={
            "bill_amount": "avg_bill",
            "patient_id": "patient_count"
        }
    )

    df = df.sort_values(by="specialty", ascending=True)

    return df[["specialty", "avg_bill", "patient_count"]]