FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Install required dependencies
RUN pip install pandas pyarrow scikit-learn==1.5.0

# Copy the scoring script
COPY score.py .

# Run the script for May 2023
CMD ["python", "score.py", "--year", "2023", "--month", "5"]