import argparse
import pickle
import pandas as pd
import numpy as np

def read_data(filename):
    """Read and preprocess the taxi data"""
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Score taxi trip duration predictions')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2023)')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
    
    args = parser.parse_args()
    
    year = args.year
    month = args.month
    
    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    # Create the data URL
    data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    
    print(f"Processing data for {year:04d}-{month:02d}")
    print(f"Data URL: {data_url}")
    
    # Read and process the data
    df = read_data(data_url)
    
    # Create ride_id
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    # Make predictions
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Calculate and print statistics
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    
    print(f"Mean predicted duration: {mean_pred:.2f} minutes")
    print(f"Standard deviation: {std_pred:.2f} minutes")
    print(f"Number of predictions: {len(y_pred)}")
    
    # Create results dataframe
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    
    # Save results
    output_file = f'predictions_{year:04d}_{month:02d}.parquet'
    df_result.to_parquet(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return mean_pred

if __name__ == '__main__':
    main()