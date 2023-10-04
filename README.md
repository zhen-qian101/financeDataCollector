# financeDataCollector

This project is a practice on collecting financial data via SEC XBRL data APIs. The workflow is shown below. The script only works for banks at the moment. i.e., only support tickers of the public listed banks.

<img width="640" alt="workflow" src="https://github.com/zhen-qian101/financeDataCollector/assets/90771509/d866a3b6-4a1f-48a4-b9b4-e927865e4617">

The main challenge of this project is that the financial data is not standardised. 
- Data of the same category are stored in different tags for different companies.
- For one company, it store the data (i.e., net income values) with different tags across different years.
- Some company follow the GAAP accounting standard while others go with IFRS. Tags under which the data is stored are different with respect to different accounting standards.

This script should work for most publicly listed banks but is not 100% bullet proofed.


**How to run the script?**

**Step 1**: create an environment and install dependancies.

pip install -r requirements.txt

**Step 2**: run the script from command line, enter stocker ticker and stock price, and the metrics will be calculated and displayed on screen.

inputs:

<img width="640" alt="workflow" src="https://github.com/zhen-qian101/financeDataCollector/assets/90771509/9187c9f6-1369-4371-9a24-3d3feb717641">

outputs:

<img width="640" alt="workflow" src="https://github.com/zhen-qian101/financeDataCollector/assets/90771509/aa058b12-ca40-49c5-8e9d-764575641cff">
