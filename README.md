### Predictive Analysis Project
- [Link to project description](https://kaggle.com/c/rossmann-store-sales/data)
- [Link to planning notes](https://drive.google.com/drive/folders/1LPEJu1_YDCzkdOEA4zpZfkf7RXBGHftR)

### Workflow
- Everybody works on their own branch
- Add a test in `src/tests.py`. It should not be extra work, just copy the code you used to develop whatever you did there
    - For a trained model, add a test that loads that model
    - For a new forecaster, just import the forecaster and add it to `TestTraining.Models`
- Before you merge something to master, you run at least the new tests with `python src/main.py test`,
 and if you deleted a test because it takes to much time, please add it afterwards again

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