import pandas as pd

# Load the CSV file
df = pd.read_csv("../scaler_fits_files_info.csv")

# Select every 40th row and extract the NOAAID column
noaaid_values = df['NOAAID'][::40].reset_index(drop=True)

# Display the extracted NOAAID values
# Save the extracted NOAAID values to a new CSV file
noaaid_values.to_csv("extracted_noaaid.csv", index=False, header=["NOAAID"])

print("NOAAID values from every 40th row have been saved to 'extracted_noaaid.csv'")