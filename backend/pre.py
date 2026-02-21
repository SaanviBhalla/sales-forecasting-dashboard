import pandas as pd

# loading the dataset 
df = pd.read_csv("uploads/sales.csv")

# convert date format to datetime
df["Date"] = pd.to_datetime(df["Date"])
df = df.drop(columns=["Unnamed: 0"])

# aggregation by date and product category
sales = (
    df.groupby(["Date", "Product Category"])["Quantity"]
      .sum()
      .reset_index()
      .rename(columns={"Quantity": "Sales"})
)

# turns missing dates into continous time series 
full_dates = pd.date_range(sales["Date"].min(), sales["Date"].max())

categories = sales["Product Category"].unique()

full_index = pd.MultiIndex.from_product(
    [full_dates, categories],
    names=["Date", "Product Category"]
)

sales = (
    sales.set_index(["Date", "Product Category"])
       .reindex(full_index, fill_value=0)
       .reset_index()
)

# categorizes sales into previous day, same day last week, recent trends
sales = sales.sort_values(["Product Category", "Date"])

sales["yesterday_sales"] = sales.groupby("Product Category")["Sales"].shift(1)
sales["last_week_sales"] = sales.groupby("Product Category")["Sales"].shift(7)

sales["week_sales_mean"] = (
    sales.groupby("Product Category")["Sales"]
         .shift(1)
         .rolling(7)
         .mean()
)

sales = sales.dropna()

# turn this new processed dataset into a csv file 
sales.to_csv("artifacts/pre_sales.csv")
