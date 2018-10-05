#### Enumeration for features so they are easily accessible ####

STORE = 0
DAY_OF_WEEK = 1
PROMO = 2
STATE_HOLIDAY = 3
SCHOOL_HOLIDAY = 4
YEAR = 5
MONTH = 6
DAY = 7
WEEK_OF_YEAR = 8
STORE_TYPE = 9
ASSORTMENT = 10
COMPETETION_DISTANCE = 11
PROMO2 = 12
AVG_SALES = 13
AVG_CUSTOMER = 14

FEATURES = ['Store', 'DayOfWeek', 'Promo',
            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
            'CompetitionDistance', 'Promo2', 'Year', 'Month', 'Open', 'Day']  #

ONE_HOT_FEATURES = {"StoreType": ['ST1', 'ST2', 'ST3', 'ST4'], "Assortment": ["AS1", "AS2", "AS3"],
                    "StateHoliday": ["SH1", "SH2", "SH3", "SH4"],
                    "DayOfWeek": ["DW1", "DW2", "DW3", "DW4", "DW5", "DW6", "DW7"]}

feature_mat = list(ONE_HOT_FEATURES.values())
_f = [val for sublist in feature_mat for val in sublist]
FEATURES.extend(_f)

DROP_FEATURES = ["StoreType", "StateHoliday", "Assortment", "DayOfWeek"]
for _d in DROP_FEATURES:
    FEATURES.remove(_d)

FEATURE_COUNT = len(FEATURES)
abcd = {
    "a": [1, 0, 0, 0],
    "b": [0, 1, 0, 0],
    "c": [0, 0, 1, 0],
    "d": [0, 0, 0, 1]
}
