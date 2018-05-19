### Predictive Analysis Project
- [Link to project description](kaggle.com/c/rossmann-store-sales/data)
- [Link to planning notes](https://drive.google.com/drive/folders/1LPEJu1_YDCzkdOEA4zpZfkf7RXBGHftR)

### Project Structure
- `/`
    - README
    - requirements.txt
- `data` contains the data.zip
- `src` code
    - `main.py` CLI for the app
    - `data.py` class Data
        - unzips /data/data.zip if not done yet
        - Reads csv data to pandas
        - Creates `tf.placeholders` for data batches
        - Splits data into train and test set
        - Provides a method to fetch the next train data batch
        - Estimates missing data
    - `visualize.py` All visualizations
    - `forecaster.py` Interface for the forecaster
    - `evaluate.py` Backtesting module

### Setup
1. Clone this project
2. Create a virtual environment and install requirements
With Anaconda and Python3:

    ```
    conda create --name irs
    source activate irs
    pip install -r requirements.txt
    ```

3. Start app `python src/main.py <command>`