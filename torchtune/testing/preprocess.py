import pandas as pd

def preprocess_sample(row):
    positive_changes = []
    negative_changes = []
    
    for i in range(1, 501):  # Adjust the range according to your number of companies
        company_key = f"Company{i}"
        stock_change = row[company_key]
        if pd.notna(stock_change):  # Ensuring there is data to process
            change_description = f"{company_key} {stock_change:+.2f}%"
            if stock_change > 0:
                positive_changes.append(change_description)
            elif stock_change < 0:
                negative_changes.append(change_description)
    
    row['positive_changes'] = ', '.join(positive_changes)
    row['negative_changes'] = ', '.join(negative_changes)
    return row
