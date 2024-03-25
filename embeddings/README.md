# Embedding Distance Calculation

This Python script calculates the Euclidean distance and cosine similarity between two string embeddings. It utilizes an API to generate the embeddings for the provided strings.

## Dependencies

- Python 3.x
- argparse
- os
- json
- requests
- numpy
- scikit-learn

## Usage

1. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the script with the desired arguments:

   ```bash
   python embed_dist.py <string1> <string2>
   ```

   - `<string1>`: The first string to calculate the embedding for.
   - `<string2>`: The second string to calculate the embedding for.

3. The script will send a request to the specified API endpoint to generate the embeddings for the provided strings.

4. If the API call is successful, the script will calculate the Euclidean distance and cosine similarity between the embeddings and display the results.

5. If the API call fails or returns a non-200 status code, an error message will be displayed.

## Example

```bash
python embed_dist.py "This is the first string." "This is the second string."
```

This command will calculate the Euclidean distance and cosine similarity between the embeddings of the two provided strings.

## Notes

- The script assumes the existence of an API endpoint that accepts a JSON payload containing the strings and returns the corresponding embeddings.
- Replace the API endpoint URL (`'https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/'`) with your actual API endpoint.
- The script expects the API response to be a JSON object containing an `'embedding'` field, which should be a list of embeddings for the provided strings.
- Adjust the API request payload and response handling based on your specific API requirements.
- The script uses the scikit-learn library to calculate the cosine similarity between the embeddings.

Feel free to modify the script to suit your specific use case and API integration requirements.
