import pandas as pd


# Read CSV file
df = pd.read_csv('C:/Users/vuong/Python/PART 2/customer_table (1).csv')
dc = pd.read_csv('C:/Users/vuong/Python/PART 2/market_trend_table.csv')

# Remove rows with missing values
new_df = df.dropna()
new_dc = dc.dropna()


# Remove duplicate rows
df_no_duplicates = df.drop_duplicates()
dc_no_duplicates = dc.drop_duplicates()

# Remove column 'CustomerID'and 'ProductGroupID'
to_drop = ['CustomerID']
df.drop(to_drop, inplace=True, axis=1)

to_drop1 = ['MarketTrendID','ProductGroupID']
dc.drop(to_drop1, inplace=True, axis=1)

# Remove all non-numeric characters in the 'PhoneNumber' column
df['PhoneNumber'] = df['PhoneNumber'].str.replace(r'[^0-9]', '', regex=True)


# Rename column
df.rename(columns={'PhoneNumber': 'Phone Number'}, inplace=True)
df.rename(columns={'FirstName': 'First Name'}, inplace=True)
df.rename(columns={'LastName': 'Last Name'}, inplace=True)
df.rename(columns={'PostalCode': 'Postal Code'}, inplace=True)
df.rename(columns={'DateOfBirth': 'Date Of Birth'}, inplace=True)

# Print final result
print(df)
print(dc)
