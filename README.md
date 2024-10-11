# DataCleaningTeamBravoLifebear
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    filename='data_cleaning.log',
    filemode='w',  # Overwrite the log file each time
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the path to your CSV file
csv_file_path = '/content/lifebear.csv'  # Update this path as needed

# Define paths for cleaned data and garbage data
cleaned_csv_path = '/content/lifebear_cleaned.csv'  # Update as needed
garbage_csv_path = '/content/lifebear_garbage.csv'  # Update as needed

# Function to count total rows (excluding header)
def count_total_rows(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            total = sum(1 for line in f) - 1  # Subtract 1 for header
        return total
    except Exception as e:
        logging.error(f"Error counting rows: {e}")
        raise

# Function to clean each chunk
def clean_chunk(df):
    try:
        original_count = len(df)
        
        # 1. Handle Missing Data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fill numeric columns with mean
        df_numeric_filled = df[numeric_cols].fillna(df[numeric_cols].mean())
        # Fill categorical columns with 'Unknown'
        df_categorical_filled = df[categorical_cols].fillna('Unknown')
        
        # Combine filled data
        df = pd.concat([df_numeric_filled, df_categorical_filled], axis=1)
        
        # 2. Remove Duplicates
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_after = df.duplicated().sum()
        duplicates_removed = duplicates_before - duplicates_after
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate rows")
        
        # 3. Convert Data Types
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if 'date_column' in df.columns:
            df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
        
        # After conversion, handle new missing values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        # 4. Handle Outliers using IQR for a specific column, e.g., 'salary'
        garbage_chunk = pd.DataFrame()
        if 'salary' in df.columns:
            Q1 = df['salary'].quantile(0.25)
            Q3 = df['salary'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]
            num_outliers = len(outliers)
            df = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]
            if num_outliers > 0:
                logging.info(f"Removed {num_outliers} outlier rows based on 'salary'")
                garbage_chunk = pd.concat([garbage_chunk, outliers], axis=0)
        else:
            logging.warning("'salary' column not found for outlier removal")
        
        # 5. Rename Columns for Consistency
        if 'Name' in df.columns:
            df = df.rename(columns={'Name': 'name'})
            logging.info("Renamed column 'Name' to 'name'")
        
        # 6. Standardize Data
        if 'name' in df.columns:
            df['name'] = df['name'].str.lower().str.strip()
        
        # 7. Deal with Invalid Data
        if 'age' in df.columns:
            before_invalid = df[~df['age'].between(0, 120)]
            num_invalid_age = len(before_invalid)
            if num_invalid_age > 0:
                logging.info(f"Removed {num_invalid_age} rows with invalid 'age'")
                df = df[df['age'].between(0, 120)]
                garbage_chunk = pd.concat([garbage_chunk, before_invalid], axis=0)
        
        cleaned_count = len(df)
        logging.info(f"Cleaned chunk: {original_count} original rows, {cleaned_count} cleaned rows")
        
        return df, garbage_chunk  # Return cleaned data and garbage as separate DataFrames
    except Exception as e:
        logging.error(f"Error cleaning chunk: {e}")
        raise

# Function to process the CSV in specified number of chunks
def process_csv_in_chunks(file_path, num_chunks, delimiter=';'):
    try:
        total_rows = count_total_rows(file_path)
        chunk_size = total_rows // num_chunks
        logging.info(f"Processing CSV in {num_chunks} chunks of {chunk_size} rows each")
        print(f"Processing CSV in {num_chunks} chunks of {chunk_size} rows each")
        
        cleaned_chunks = []
        
        # Initialize garbage CSV: write header
        with open(garbage_csv_path, 'w', encoding='utf-8') as f_garbage:
            # Read the first chunk to get headers
            first_chunk = pd.read_csv(file_path, delimiter=delimiter, nrows=chunk_size)
            f_garbage.write(';'.join(first_chunk.columns) + '\n')
            del first_chunk  # Free memory
        
        # Initialize the reader
        reader = pd.read_csv(file_path, delimiter=delimiter, chunksize=chunk_size, low_memory=False)
        
        for i, chunk in enumerate(reader, 1):
            logging.info(f"Processing chunk {i}/{num_chunks}")
            print(f"Processing chunk {i}/{num_chunks}")
            try:
                cleaned_chunk, garbage_chunk = clean_chunk(chunk)
                cleaned_chunks.append(cleaned_chunk)
                
                # Append garbage rows to garbage CSV
                if not garbage_chunk.empty:
                    garbage_chunk.to_csv(
                        garbage_csv_path,
                        mode='a',
                        header=False,
                        index=False,
                        sep=';'
                    )
                    logging.info(f"Appended {len(garbage_chunk)} garbage rows from chunk {i}")
                
            except Exception as e:
                logging.error(f"Failed to process chunk {i}: {e}")
        
        # Handle any remaining rows if total_rows is not divisible by num_chunks
        remainder = total_rows % num_chunks
        if remainder != 0:
            logging.info(f"Processing remaining {remainder} records")
            print(f"Processing remaining {remainder} records")
            try:
                last_chunk = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    skiprows=range(1, num_chunks * chunk_size + 1),
                    nrows=remainder,
                    header=None,
                    names=chunk.columns
                )
                cleaned_last_chunk, garbage_last_chunk = clean_chunk(last_chunk)
                cleaned_chunks.append(cleaned_last_chunk)
                
                # Append garbage rows to garbage CSV
                if not garbage_last_chunk.empty:
                    garbage_last_chunk.to_csv(
                        garbage_csv_path,
                        mode='a',
                        header=False,
                        index=False,
                        sep=';'
                    )
                    logging.info(f"Appended {len(garbage_last_chunk)} garbage rows from the remaining chunk")
                
            except Exception as e:
                logging.error(f"Failed to process remaining chunk: {e}")
        
        # Remerge all cleaned chunks into a single DataFrame
        df_cleaned = pd.concat(cleaned_chunks, ignore_index=True)
        df_cleaned.reset_index(drop=True, inplace=True)
        
        logging.info("Completed processing all chunks")
        print("Completed processing all chunks")
        
        return df_cleaned
    except Exception as e:
        logging.error(f"Error processing CSV in chunks: {e}")
        raise

# Execute the processing
df_cleaned = process_csv_in_chunks(csv_file_path, 10)

# Display the first few rows of the cleaned DataFrame
logging.info("Displaying the first few rows of the cleaned DataFrame")
print("\nCleaned DataFrame Head:")
print(df_cleaned.head())

# Display summary statistics
logging.info("Displaying summary statistics of the cleaned DataFrame")
print("\nSummary Statistics:")
print(df_cleaned.describe())

# Display missing values after cleaning
logging.info("Displaying missing values after cleaning")
print("\nMissing Values After Cleaning:")
print(df_cleaned.isnull().sum())

# Save the cleaned data to a new CSV file
try:
    df_cleaned.to_csv(cleaned_csv_path, index=False, sep=';')
    logging.info(f"Cleaned data saved to {cleaned_csv_path}")
    print(f"\nCleaned data saved to {cleaned_csv_path}")
except Exception as e:
    logging.error(f"Error saving cleaned data: {e}")
    print(f"\nError saving cleaned data: {e}")
