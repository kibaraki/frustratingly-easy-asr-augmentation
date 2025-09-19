import pandas as pd
import os

def split_data(
    input_dir: str = 'csv', output_dir: str = 'split', train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1
):

    combined_df = pd.read_csv('extracted_data.csv')
    print(f"Found {len(combined_df)} total rows.")


    # .sample(frac=1) shuffles the rows randomly
    # .reset_index(drop=True) creates a clean new index for the shuffled data
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    print("Shuffled data successfully.")

    total_rows = len(shuffled_df)
    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)

    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]

    print(f"Split complete: Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)})")

    #os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    #os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    #os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    train_df.to_csv(('train70.csv'), index=False)
    val_df.to_csv(('val15.csv'), index=False)
    test_df.to_csv(( 'test15.csv'), index=False)
    
    print(f"Files saved successfully in '{output_dir}' directory.")


if __name__ == "__main__":

    split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
