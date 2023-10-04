from sec_cik_mapper import StockMapper
import requests
import numpy as np
import pandas as pd
from datetime import date

def stock_data(ticker):

    print('fetching data ...')
    
    agent = {
    'User-Agent':'s3888611@student.rmit.edu.au'
        }
    
    mapper = StockMapper()
    try:
        mapper.ticker_to_cik[ticker]
    except:
        print("Please enter a valid ticker")
    CIKraw=mapper.ticker_to_cik[ticker]
    CIK=str(CIKraw)
    url_basic='https://data.sec.gov/api/xbrl/companyfacts/CIK'
    url=url_basic+CIK+'.json'
    try:
        raw=requests.get(url,headers=agent).json()
    except:
        print('The SEC has not returned the database for this stock')

    return raw



tags_mapping = {

    #Assets
    'cash':['CashAndDueFromBanks', 'CashAndBankBalancesAtCentralBanks', 'CashAndCashEquivalents', 'CashAndBankBalancesAtCentralBanks'],
    'deposits_with_banks':['InterestBearingDepositsInBanks', 'LoansAndAdvancesToBanks'],
    'reverse_repurchase_agreement':['FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell', 'ReverseRepurchaseAgreementsAndCashCollateralOnSecuritiesBorrowed'],
    'borrowed_securities':['SecuritiesBorrowed'],
    'st_trading_securities':['TradingSecurities', 'FinancialAssetsAtFairValueThroughProfitOrLossClassifiedAsHeldForTrading'],
    'st_derivative_assets':['DerivativeAssets', 'DerivativeFinancialAssets', 'DerivativeFinancialAssetsHeldForTrading'],
    'lt_available_for_sale_assets':['DebtSecuritiesAvailableForSaleExcludingAccruedInterest'],
    'lt_held_to_maturity_assets':['DebtSecuritiesHeldToMaturityExcludingAccruedInterestAfterAllowanceForCreditLoss'],
    'allowance_for_loan_losses':['FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest', 'FinancingReceivableAllowanceForCreditLosses', 'LoansAndLeasesReceivableAllowance'],
    'loans':['FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss', 'NotesReceivableNet', 'LoansAndLeasesReceivableNetReportedAmount', 'LoansReceivableNet', 'LoansAndAdvancesToCustomers'],
    'ppe':['PropertyPlantAndEquipmentNet'],
    'goodwill':['Goodwill', 'IntangibleAssetsAndGoodwill'],
    'other_intangible_asset':['IntangibleAssetsNetExcludingGoodwill', 'OtherIntangibleAssetsNet'],
    'other_assets':['OtherAssets'],
    'other_financial_assets':['OtherFinancialAssets'],
    'other_nonfinancial_assets':['OtherNonfinancialAssets'],
    'total_assets':['Assets'],

    #Liability
    'us_deposits_non_interestbearing':['NoninterestBearingDepositLiabilitiesDomestic'],
    'us_deposits_interestbearing':['InterestBearingDepositLiabilitiesDomestic'],
    'foreign_deposits_non_interestbearing':['NoninterestBearingDepositLiabilitiesForeign'],
    'foreign_deposits_interestbearing':['InterestBearingDepositLiabilitiesForeign'],
    'total_deposits':['Deposits', 'DepositsFromCustomers'],
    'repurchase_agreement':['FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase', 'SecuritiesSoldUnderAgreementsToRepurchase', 'RepurchaseAgreementsAndCashCollateralOnSecuritiesLent'],
    'trading_liabilities':['TradingLiabilities', 'FinancialLiabilitiesAtFairValueThroughProfitOrLossClassifiedAsHeldForTrading', 'DerivativeFinancialLiabilitiesHeldForTrading'],
    'short_term_debt':['ShortTermBorrowings', 'OtherShortTermBorrowings', 'ShorttermDebtFairValue', 'FinancialLiabilitiesAtFairValueThroughProfitOrLossDesignatedAsUponInitialRecognition'], 
    'accounts_payable':['AccountsPayableAndAccruedLiabilitiesCurrentAndNoncurrent', 'AccruedLiabilitiesAndOtherLiabilities', 'OtherAccountsPayableAndAccruedLiabilities'],
    'long_term_debt':['LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities', 'LongTermDebt', 'UnsecuredLongTermDebt', 'DebtSecurities'],
    'other_liabilities':['OtherLiabilities'],
    'total_liabilities':['Liabilities'],
    'total_stockholders_equity':['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'Equity'],

    #income
    'revenues':['Revenues', 'RevenuesNetOfInterestExpense', 'InterestAndDividendIncomeOperating', 'RevenueAndOperatingIncome', 'InterestIncomeOperating', 'RevenueFromInterest'],
    'net_income':['NetIncomeLoss', 'ProfitLoss', 'ProfitLossFromContinuingOperations'],
    'eps':['EarningsPerShareDiluted', 'DilutedEarningsLossPerShare', 'DilutedEarningsLossPerShare']

}


def tag_finder(stock_data, accounting_standard, currency, target_value):
    possible_tags = []
    accounts = list(stock_data['facts'][accounting_standard].keys())
    for acc in accounts:
        units = list(stock_data['facts'][accounting_standard][acc]['units'].keys())
        if units[0] == currency:
            records = stock_data['facts'][accounting_standard][acc]['units'][currency]
            for rec in records:
                val = rec['val']
                if val == target_value:
                    possible_tags.append(acc)
    possible_tags = list(set(possible_tags))
    return possible_tags


def catalog_type(data):
    """
    checks for catalog type
    """

    catalog = []

    for item in data['facts']:
        if item.find('us-gaap')!=-1:
            catalog.append('us-gaap')
        elif item.find('ifrs-full')!=-1:
            catalog.append('ifrs-full')
    
    catalog.sort(reverse=True)
            
    if len(catalog) == 0:
        catalog.append('Catalog not recognized! Check site https://data.sec.gov/api/xbrl/companyfacts/CIK for either us-gaap or ifrs-full')
    
    return catalog


         
def get_latest_period(stock_data, item_name: str, tags_mapping=tags_mapping):

    #item_name must be one of the keys in the tags_mapping

    catalog = catalog_type(stock_data)
    if len(catalog) == 0:
        return catalog

    #find the latest period
    latest_period = None
    latest_period_dict = {}
    for expression in tags_mapping[item_name]:

        for acc_rule in catalog:
            if expression in stock_data['facts'][acc_rule]:
                currency = list(stock_data['facts'][acc_rule][expression]['units'].keys())
                for cur in currency:
                    records = stock_data['facts'][acc_rule][expression]['units'][cur]
                    for rec in list(reversed(records)):
                        if 'frame' in rec.keys():
                            latest_period_dict[cur] = rec['frame']
                            if len(latest_period_dict[cur]) > 6:
                                latest_period_dict[cur] = latest_period_dict[cur][2:8]
                                break
                            if len(latest_period_dict[cur]) == 6:
                                latest_period_dict[cur] = latest_period_dict[cur][2:6]
                                break


        if len(latest_period_dict) > 0:
            for v in latest_period_dict.values():
                if date.today().year - int(v[:4]) < 2:
                    latest_period = v

    if not latest_period:
        if len(latest_period_dict) > 0:
            latest_period = 'no latest data'

    return latest_period


def get_item_values(stock_data, item_name: str, statement_type='balancesheet', n_period=4, tags_mapping=tags_mapping) -> np.ndarray:

    #item_name must be one of the keys in the tags_mapping

    latest_period = get_latest_period(stock_data, item_name)
    if not latest_period:
        item_values = np.array([f"{item_name} data is unavailable for" + " " + stock_data['entityName']])
        return item_values
    
    if latest_period == 'no latest data':
        item_values = np.array([f"no latest {item_name} data for" + " " + stock_data['entityName']])
        return item_values

    catalog = catalog_type(stock_data)
    if len(catalog) == 0:
        return catalog
    
    item_values = []

    if statement_type == 'balancesheet':

        for i in range(n_period):

            item_value = pd.Series()

            for acc_rule in catalog:
                if acc_rule == 'us-gaap':
                    period_temp = str(pd.Period(latest_period)-i)
                    frame_temp = f'CY{period_temp}I'
                elif acc_rule == 'ifrs-full':
                    period_temp = str(pd.Period(latest_period)-2*i)
                    frame_temp = f'CY{period_temp}I'                   


                for expression in tags_mapping[item_name]:
                    if expression in stock_data['facts'][acc_rule]:
                        currency = list(stock_data['facts'][acc_rule][expression]['units'].keys())
                        for cur in currency:
                            data_df = pd.DataFrame.from_dict(stock_data['facts'][acc_rule][expression]['units'][cur])
                            item_value = data_df[data_df['frame'] == frame_temp]['val']


                    if len(item_value) > 0:
                        item_values.append(item_value.values[0])
                        break

        if len(item_values) == 0:
            item_values = np.array([f"{item_name} data is unavailable for" + " " + stock_data['entityName']])
        else:
            item_values = np.array(item_values)


    if statement_type == 'incomestatement':

        period = int(n_period/4)

        if len(latest_period) == 6:
            latest_period = int(latest_period[:4])-1

        for i in range(period):
            period_temp = int(latest_period) - i
            frame_temp = f'CY{period_temp}'
            item_value = pd.Series()

            for acc_rule in catalog:

                for expression in tags_mapping[item_name]:                
                    if expression in stock_data['facts'][acc_rule]:
                        currency = list(stock_data['facts'][acc_rule][expression]['units'].keys())
                        for cur in currency:
                            data_df = pd.DataFrame.from_dict(stock_data['facts'][acc_rule][expression]['units'][cur])
                            item_value = data_df[data_df['frame'] == frame_temp]['val']


                    if len(item_value) > 0:
                        item_values.append(item_value.values[0])
                        break

        if len(item_values) == 0:
            item_values = np.array([f"{item_name} data is unavailable for" + " " + stock_data['entityName']])
        else:
            item_values = np.array(item_values)


    return item_values

def create_asset_dict(stock_data):

    asset_dict = {

        'Cash':get_item_values(stock_data, 'cash')[0],
        'Deposits with banks':get_item_values(stock_data, 'deposits_with_banks')[0],
        'Reverse repurchase agreement':get_item_values(stock_data, 'reverse_repurchase_agreement')[0],
        'Borrowed securities':get_item_values(stock_data, 'borrowed_securities')[0],
        'Short term trading securities':get_item_values(stock_data, 'st_trading_securities')[0],
        'Short term derivative assets':get_item_values(stock_data, 'st_derivative_assets')[0],
        'Long term available-for-sale assets':get_item_values(stock_data, 'lt_available_for_sale_assets')[0],
        'Long term held-to-maturity assets':get_item_values(stock_data, 'lt_held_to_maturity_assets')[0],
        'Loans':get_item_values(stock_data, 'loans')[0],
        'Allowance for loan losses':get_item_values(stock_data, 'allowance_for_loan_losses')[0],
        'Property plant and equipment':get_item_values(stock_data, 'ppe')[0],
        'Goodwill':get_item_values(stock_data, 'goodwill')[0],
        'Other intangible':get_item_values(stock_data, 'other_intangible_asset')[0],
        'Other financial assets': get_item_values(stock_data, 'other_financial_assets')[0],
        'Other non-financial assets': get_item_values(stock_data, 'other_nonfinancial_assets')[0],
        'Other assets': get_item_values(stock_data, 'other_assets')[0],
        'Unclassified assets':None,
        'Total assets':get_item_values(stock_data, 'total_assets')[0]

    }

    unclassified_assets = asset_dict['Total assets']

    if type(unclassified_assets) != np.int64:
        
        return unclassified_assets


    for k,v in asset_dict.items():
        if k != 'Allowance for loan losses' and k != 'Total assets' and k != 'Unclassified assets' and type(v) == np.int64:
            unclassified_assets = unclassified_assets - v

    asset_dict['Unclassified assets'] = unclassified_assets    

    return asset_dict

def report_assets(stock_data):

    assets = create_asset_dict(stock_data)
    if type(assets) != str:
        for k,v in assets.items():
            print(k+':', v)
    else:
        print(assets)



def create_liability_dict(stock_data):

    liability_dict = {

        'US deposits non-interestbearing':get_item_values(stock_data, 'us_deposits_non_interestbearing')[0],
        'US deposits interestbearing':get_item_values(stock_data, 'us_deposits_interestbearing')[0],
        'Foreign deposits non-interestbearing':get_item_values(stock_data, 'foreign_deposits_non_interestbearing')[0],
        'Foreign deposits interestbearing':get_item_values(stock_data, 'foreign_deposits_interestbearing')[0],
        'Total deposits': get_item_values(stock_data, 'total_deposits')[0],
        'Repurchase agreement':get_item_values(stock_data, 'repurchase_agreement')[0],
        'Trading liabilities':get_item_values(stock_data, 'trading_liabilities')[0],
        'Short term debt':get_item_values(stock_data, 'short_term_debt')[0],
        'Accounts payable':get_item_values(stock_data, 'accounts_payable')[0],
        'Long term debt':get_item_values(stock_data, 'long_term_debt')[0],
        'Other liabilities': get_item_values(stock_data, 'other_liabilities')[0],
        'Unclassified liabilities':None,
        'Total liabilities':get_item_values(stock_data, 'total_liabilities')[0],
        'Total stockholders equity':get_item_values(stock_data, 'total_stockholders_equity')[0]

    }

    unclassified_liabilities = liability_dict['Total liabilities']

    if type(unclassified_liabilities) != np.int64:
        
        return unclassified_liabilities


    for k, v in liability_dict.items():
        if k in ['Total deposits', 'Repurchase agreement', 'Trading liabilities', 'Short term debt', 'Accounts payable', 'Long term debt', 'Other liabilities'] and type(v) == np.int64: 
            unclassified_liabilities = unclassified_liabilities - v

    liability_dict['Unclassified liabilities'] = unclassified_liabilities    

    return liability_dict

def report_liabilities(stock_data):
    
    liabilities = create_liability_dict(stock_data)
    if type(liabilities) != str:
        for k,v in liabilities.items():
            print(k+':', v)
    else:
        print(liabilities)



def extract_attributes_for_visualization_assets(stock_data):

    asset_dict = create_asset_dict(stock_data)
    new_dict = {}

    for k,v in asset_dict.items():
        if type(v) == np.int64 and k not in ['Allowance for loan losses', 'Total assets']:
            new_dict[k] = v

    return new_dict


def extract_attributes_for_visualization_liabilities(stock_data):

    liability_dict = create_liability_dict(stock_data)
    new_dict = {}

    for k,v in liability_dict.items():
        if type(v) == np.int64 and k not in ['US deposits non-interestbearing', 'US deposits interestbearing', 'Foreign deposits non-interestbearing', 'Foreign deposits interestbearing', 'Total liabilities']:
            new_dict[k] = v

    return new_dict

# def draw_asset_piechart(stock_data):

#     data_dict =  extract_attributes_for_visualization_assets(stock_data)
    
#     labels = list(data_dict.keys())
#     values = list(data_dict.values())
#     asset_distribution_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3,textinfo='label+percent', insidetextorientation='radial')])

#     return asset_distribution_pie_chart

# def draw_liability_piechart(stock_data):

#     data_dict =  extract_attributes_for_visualization_liabilities(stock_data)
    
#     labels = list(data_dict.keys())
#     values = list(data_dict.values())
#     liability_distribution_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3,textinfo='label+percent', insidetextorientation='radial')])

#     return liability_distribution_pie_chart

def validate_value(input_value):

    if type(input_value) == np.str_:
        output_value = 0
    else:
        output_value = input_value

    return output_value


def A_loss_on_mortgage_by_mortgage(stock_data):

    loan_losses = get_item_values(stock_data, 'allowance_for_loan_losses')[0]
    loan_losses = validate_value(loan_losses)
    loans = get_item_values(stock_data, 'loans')[0]
    loans = validate_value(loans)

    if loans != 0:
        loss_on_mortgage_by_mortgage = round(loan_losses/loans, 4)
    else:
        loss_on_mortgage_by_mortgage = np.str_('Error: divided by zero')

    return loss_on_mortgage_by_mortgage

def B_loss_on_mortgage_by_total_assets(stock_data):

    loan_losses = get_item_values(stock_data, 'allowance_for_loan_losses')[0]
    loan_losses = validate_value(loan_losses)
    total_assets = get_item_values(stock_data, 'total_assets')[0]
    total_assets = validate_value(total_assets)

    if total_assets != 0:
        loss_on_mortgage_by_total_assets = round(loan_losses/total_assets, 4)
    else:
        loss_on_mortgage_by_total_assets = np.str_('Error: divided by zero')    

    return loss_on_mortgage_by_total_assets



def C_loan_to_deposit(stock_data):

    loans = get_item_values(stock_data, 'loans')[0]
    loans = validate_value(loans)
    total_deposits = get_item_values(stock_data, 'total_deposits')[0]
    total_deposits = validate_value(total_deposits)
    trading_liabilities = get_item_values(stock_data, 'trading_liabilities')[0]
    trading_liabilities = validate_value(trading_liabilities)

    if (total_deposits+trading_liabilities) != 0:
        loan_to_deposit = round(loans/(total_deposits+trading_liabilities),4)
    else:
        loan_to_deposit = np.str_('Error: divided by zero')     

    return loan_to_deposit

def D_debt_to_deposit(stock_data):

    short_term_debt = get_item_values(stock_data, 'short_term_debt')[0]
    short_term_debt = validate_value(short_term_debt)
    long_term_debt = get_item_values(stock_data, 'long_term_debt')[0]
    long_term_debt = validate_value(long_term_debt)
    trading_liabilities = get_item_values(stock_data, 'trading_liabilities')[0]
    trading_liabilities = validate_value(trading_liabilities)
    total_debt = short_term_debt + long_term_debt + trading_liabilities
    total_debt = validate_value(total_debt)
    total_deposits = get_item_values(stock_data, 'total_deposits')[0]
    total_deposits = validate_value(total_deposits)

    if total_deposits != 0:
        debt_to_deposit = round(total_debt/total_deposits, 4)
    else:
        debt_to_deposit = np.str_('Error: divided by zero')      
    
    return debt_to_deposit

def E_debt_to_equity(stock_data):

    short_term_debt = get_item_values(stock_data, 'short_term_debt')[0]
    short_term_debt = validate_value(short_term_debt)
    long_term_debt = get_item_values(stock_data, 'long_term_debt')[0]
    long_term_debt = validate_value(long_term_debt)
    trading_liabilities = get_item_values(stock_data, 'trading_liabilities')[0]
    trading_liabilities = validate_value(trading_liabilities)    
    total_debt = short_term_debt + long_term_debt + trading_liabilities
    total_debt = validate_value(total_debt)
    stockholders_equity = get_item_values(stock_data, 'total_stockholders_equity')[0]
    stockholders_equity = validate_value(stockholders_equity)

    if stockholders_equity != 0:
        debt_to_equity = round(total_debt/stockholders_equity, 4)
    else:
        debt_to_equity = np.str_('Error: divided by zero')          
    
    return debt_to_equity




def F_equity_to_deposit(stock_data):

    stockholders_equity = get_item_values(stock_data, 'total_stockholders_equity')[0]
    stockholders_equity = validate_value(stockholders_equity)
    total_deposits = get_item_values(stock_data, 'total_deposits')[0]
    total_deposits = validate_value(total_deposits)

    if total_deposits != 0:
        equity_to_deposit = round(stockholders_equity/total_deposits, 4)
    else:
        equity_to_deposit = np.str_('Error: divided by zero')     
    
    return equity_to_deposit



def metric_G(stock_data):
    
    A = A_loss_on_mortgage_by_mortgage(stock_data)
    A = validate_value(A)
    B = B_loss_on_mortgage_by_total_assets(stock_data)
    B = validate_value(B)
    D = D_debt_to_deposit(stock_data)
    D = validate_value(D)
    E = E_debt_to_equity(stock_data)
    E = validate_value(E)
    G = round(A+B+D+E, 4)

    return G

def metric_H(stock_data):

    G = metric_G(stock_data)
    E = E_debt_to_equity(stock_data)
    E = validate_value(E)

    if G != 0:
        H = round(E/G,4)
    else:
        H = np.str('Error: divided by zero')      
    return H


def J_deposits(stock_data):

    total_deposits = get_item_values(stock_data, 'total_deposits')[0]
    total_deposits = validate_value(total_deposits)
    trading_liabilities = get_item_values(stock_data, 'trading_liabilities')[0]
    trading_liabilities = validate_value(trading_liabilities)
    short_term_debt = get_item_values(stock_data, 'short_term_debt')[0]
    short_term_debt = validate_value(short_term_debt)

    cash = get_item_values(stock_data, 'cash')[0]
    cash = validate_value(cash)
    deposits_with_banks = get_item_values(stock_data, 'deposits_with_banks')[0]
    deposits_with_banks = validate_value(deposits_with_banks)
    federal_funds = get_item_values(stock_data, 'reverse_repurchase_agreement')[0]
    federal_funds = validate_value(federal_funds)
    borrowed_securities = get_item_values(stock_data, 'borrowed_securities')[0]
    borrowed_securities = validate_value(borrowed_securities)
    trading_securities = get_item_values(stock_data, 'st_trading_securities')[0]
    trading_securities = validate_value(trading_securities)
    derivative_assets = get_item_values(stock_data, 'st_derivative_assets')[0]
    derivative_assets = validate_value(derivative_assets)



    deposits = round((total_deposits + trading_liabilities + short_term_debt)/(cash + deposits_with_banks + federal_funds + borrowed_securities + trading_securities + derivative_assets), 4)

    return deposits


def loan_to_deposit_avg(stock_data, period=3):

    n_period = period * 4
    loans = get_item_values(stock_data, 'loans', statement_type='balancesheet', n_period=n_period)
    deposits = get_item_values(stock_data, 'total_deposits', statement_type='balancesheet', n_period=n_period)

    if len(loans) < n_period or len(deposits) < n_period:
        print(f'Data unavailable for {period} years. Calculate loan-to-deposit ratio with available data.')

    if len(loans) > len(deposits):
        loans = loans[:len(deposits)]
    elif len(loans) < len(deposits):
        deposits = deposits[:len(loans)]

    if type(loans[0]) != np.int64:
        loans = np.array([0])
    
    loan_to_deposit_avg = round((loans/deposits).mean(), 4)

    return loan_to_deposit_avg

def equity_debt_ratio_avg(stock_data, period=3):

    n_period = period * 4
    equity = get_item_values(stock_data, 'total_stockholders_equity', statement_type='balancesheet', n_period=n_period)
    debt = get_item_values(stock_data, 'total_liabilities', statement_type='balancesheet', n_period=n_period)

    if len(equity) < n_period or len(debt) < n_period:
        print(f'Data unavailable for {period} years. Calculate equity-to-debt ratio with available data.')

    if len(equity) > len(debt):
        equity = equity[:len(debt)]
    elif len(equity) < len(debt):
        debt = debt[:len(equity)]

    equity_debt_ratio_avg = round((equity/debt).mean(), 4)

    return equity_debt_ratio_avg

def media_equity(stock_data, period=3):

    n_period = period * 4
    equity = get_item_values(stock_data, 'total_stockholders_equity', statement_type='balancesheet', n_period=n_period)

    if len(equity) < n_period:
        print(f'Data unavailable for {period} years. Calculate media-equity with available data.')
    
    current_avg = equity.mean()
    previous_avg = equity[4:].mean()

    if current_avg > previous_avg:
        media_equity = '>'
    elif current_avg == previous_avg:
        media_equity = '='
    else:
        media_equity = '<'

    return media_equity

def avg_deposits(stock_data, period=3):

    n_period = period * 4
    deposits = get_item_values(stock_data, 'total_deposits', statement_type='balancesheet', n_period=n_period)

    if len(deposits) < n_period:
        print(f'Data unavailable for {period} years. Calculate average-deposits with available data.')
    
    current_avg = deposits.mean()
    previous_avg = deposits[4:].mean()

    if current_avg > previous_avg:
        avg_deposits = '>'
    elif current_avg == previous_avg:
        avg_deposits = '='
    else:
        avg_deposits = '<'

    return avg_deposits


def ROE_ttm(stock_data):

        equity = get_item_values(stock_data, 'total_stockholders_equity', statement_type='balancesheet', n_period=4)
        equity_avg = equity.mean()
        net_income = get_item_values(stock_data, 'net_income', statement_type='incomestatement', n_period=4)[0]

        roe = round(net_income/equity_avg, 4)

        return roe

def ROE_nperiod(stock_data, period=3):
        
        n_period = period * 4

        equity = get_item_values(stock_data, 'total_stockholders_equity', statement_type='balancesheet', n_period=n_period)
        res = np.full((period, 4), np.nan)
        res.flat[:len(equity)] = equity
        equity_avg = np.nanmean(res, axis=1)
        net_income = get_item_values(stock_data, 'net_income', statement_type='incomestatement', n_period=n_period)

        roe = round(np.nanmean(np.append((net_income/equity_avg), ROE_ttm(stock_data))), 4)

        return roe
    

def ROA_ttm(stock_data):

        asset = get_item_values(stock_data, 'total_assets', statement_type='balancesheet', n_period=4)
        asset_avg = asset.mean()
        net_income = get_item_values(stock_data, 'net_income', statement_type='incomestatement', n_period=4)[0]

        roa= round(net_income/asset_avg, 4)

        return roa

def ROA_nperiod(stock_data, period=3):
        
        n_period = period * 4

        asset = get_item_values(stock_data, 'total_assets', statement_type='balancesheet', n_period=n_period)
        res = np.full((period, 4), np.nan)
        res.flat[:len(asset)] = asset
        asset_avg = np.nanmean(res, axis=1)        

        net_income = get_item_values(stock_data, 'net_income', statement_type='incomestatement', n_period=n_period)

        roa = round(np.nanmean(np.append((net_income/asset_avg), ROA_ttm(stock_data))), 4)

        return roa


def get_shares_outstanding(stock_data):

    shares_outstanding = stock_data['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares'][-1]['val']

    return np.int64(shares_outstanding)


def PB_ttm(stock_data, stock_price):

    equity = get_item_values(stock_data, 'total_stockholders_equity', statement_type='balancesheet', n_period=4)
    equity_avg = equity.mean()

    shares_outstanding = get_shares_outstanding(stock_data)

    pb_ttm = round(stock_price*shares_outstanding/equity_avg, 4)

    return pb_ttm

def PB_nperiod(stock_data, stock_price, period=3):
        
        n_period = period * 4

        equity = get_item_values(stock_data, 'total_stockholders_equity', statement_type='balancesheet', n_period=n_period)
        res = np.full((period, 4), np.nan)
        res.flat[:len(equity)] = equity
        equity_avg = np.nanmean(res, axis=1)

        shares_outstanding = get_shares_outstanding(stock_data)

        pb = round(np.nanmean(np.append((stock_price*shares_outstanding/equity_avg), PB_ttm(stock_data, stock_price))), 4)

        return pb


def PE_ttm(stock_data, stock_price):

    eps = get_item_values(stock_data, 'eps', statement_type='incomestatement', n_period=4)[0]
    pe = round(stock_price/eps, 4)

    return pe


def PE_nperiod(stock_data, stock_price, period=1):

    n_period = period * 4

    eps = get_item_values(stock_data, 'eps', statement_type='incomestatement', n_period=n_period)
    pe = round(np.append((stock_price/eps), PE_ttm(stock_data, stock_price)).mean(), 4)

    return pe

def net_profit_margin_ttm(stock_data):

    revenues = get_item_values(stock_data, 'revenues', statement_type='incomestatement', n_period=4)[0]
    net_income = get_item_values(stock_data, 'net_income', statement_type='incomestatement', n_period=4)[0]

    net_profit_margin_ttm = round(net_income/revenues, 4)

    return net_profit_margin_ttm

def net_profit_margin_nperiod(stock_data, period=3):

    n_period = period * 4

    revenues = get_item_values(stock_data, 'revenues', statement_type='incomestatement', n_period=n_period)
    net_income = get_item_values(stock_data, 'net_income', statement_type='incomestatement', n_period=n_period)

    net_profit_margin_nperiod = round(np.append((net_income/revenues), net_profit_margin_ttm(stock_data)).mean(), 4)

    return net_profit_margin_nperiod

def EPS_10y_linear_regression_dataframe(stock_data):

    eps = get_item_values(stock_data, 'eps', statement_type='incomestatement', n_period=40)

    if len(eps) < 10:
        return f"Only {len(eps)} years' eps data available. Cannot have the linear regression of EPS of last 10 years."
    
    anniEPS=np.arange(0,len(eps))

    dictionary = {"Year": anniEPS, "EPS": eps}
    eps_df = pd.DataFrame(dictionary)

    return eps_df 



def create_metrics_dict_partI(stock_data):
    
    metrics_dict = {

        'Loss on mortgages/total mortgages':A_loss_on_mortgage_by_mortgage(stock_data),
        'Loss on mortgages/total assets':B_loss_on_mortgage_by_total_assets(stock_data),
        'Total loans/total deposits':C_loan_to_deposit(stock_data),
        'Total debt/total deposits':D_debt_to_deposit(stock_data),
        'Total debt/equity':E_debt_to_equity(stock_data),
        'Equity/Total deposits':F_equity_to_deposit(stock_data),
        'G':metric_G(stock_data),
        'H':metric_H(stock_data),
        'J':J_deposits(stock_data),

    }

    return metrics_dict

def create_metrics_dict_partII(stock_data):
    
    metrics_dict = {

        'Loan to deposit 3yr': loan_to_deposit_avg(stock_data, period=3),
        'Equity to debt 3yr': equity_debt_ratio_avg(stock_data, period=3),        
        'Media Equity 3yr': media_equity(stock_data, period=3),
        'Average Deposits 3yr': avg_deposits(stock_data, period=3),

        'Loan to deposit 5yr': loan_to_deposit_avg(stock_data, period=5),
        'Equity to debt 5yr': equity_debt_ratio_avg(stock_data, period=5),        
        'Media Equity 5yr': media_equity(stock_data, period=5),
        'Average Deposits 5yr': avg_deposits(stock_data, period=5), 

        'Loan to deposit 10yr': loan_to_deposit_avg(stock_data, period=10),
        'Equity to debt 10yr': equity_debt_ratio_avg(stock_data, period=10),
        'Media Equity 10yr': media_equity(stock_data, period=10),
        'Average Deposits 10yr': avg_deposits(stock_data, period=10)

    }

    return metrics_dict


def create_metrics_dict_partIII(stock_data, stock_price):
    
    metrics_dict = {

        'ROE TTM': ROE_ttm(stock_data),
        'ROA TTM': ROA_ttm(stock_data),
        'P/B TTM': PB_ttm(stock_data, stock_price),
        'P/E TTM': PE_ttm(stock_data, stock_price),
        'Net profit margin TTM': net_profit_margin_ttm(stock_data),

        'ROE 3yr': ROE_nperiod(stock_data, period=3),
        'ROA 3yr': ROA_nperiod(stock_data, period=3),
        'P/B 3 yr': PB_nperiod(stock_data, stock_price, period=3),
        'P/E 3yr': PE_nperiod(stock_data, stock_price, period=3),
        'Net profit margin 3yr': net_profit_margin_nperiod(stock_data, period=3),                                
        
        'ROE 5yr': ROE_nperiod(stock_data, period=5),
        'ROA 5yr': ROA_nperiod(stock_data, period=5),
        'P/B 5 yr': PB_nperiod(stock_data, stock_price, period=5),
        'P/E 5yr': PE_nperiod(stock_data, stock_price, period=5),
        'Net profit margin 5yr': net_profit_margin_nperiod(stock_data, period=5)

    }

    return metrics_dict

def report_metrics(stock_data, stock_price):

    metrics1 = create_metrics_dict_partI(stock_data)
    metrics2 = create_metrics_dict_partII(stock_data)
    metrics3 = create_metrics_dict_partIII(stock_data, stock_price)

    for k,v in metrics1.items():

        print(k+':', v)

    print('---------------------------------')

    for k,v in metrics2.items():

        print(k+':', v)

    print('---------------------------------')

    for k,v in metrics3.items():

        print(k+':', v)


def print_report(ticker, stock_price):

    data = stock_data(ticker)
    report_assets(data)
    print('--------------------------------------')
    report_liabilities(data)
    print('--------------------------------------')
    report_metrics(data, stock_price)


    


if __name__ == "__main__":

    ticker = input('Please enter a ticker\n')
    stock_price = input('Please enter stock price\n')

    print_report(ticker, float(stock_price))