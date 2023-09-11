import pandas as pd

def random_sample_csv(input_filename, output_filename, sample_size=50):
    # Read the CSV file
    df = pd.read_csv(input_filename)

    # Randomly sample the dataframe
    sampled_df = df.sample(n=sample_size, random_state=4)

    # Save the sampled dataframe to a new CSV
    sampled_df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    input_filename = '/Users/typham-swann/Downloads/data/training/test.csv' # Replace with your CSV file path
    output_filename = 'sampled_output.csv' # Name of the output CSV file
    random_sample_csv(input_filename, output_filename)
