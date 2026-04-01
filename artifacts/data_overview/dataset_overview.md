# Dataset Overview

- Number of flights: `11446`
- Number of channels: `23`
- Label counts: `{'0': 5844, '1': 5602}`
- Fold counts: `{'0': 2290, '1': 2289, '2': 2289, '3': 2289, '4': 2289}`
- Flight length summary: `{'count': 11446, 'mean': 5517.537742442775, 'std': 2211.067550453348, 'min': 605, 'q25': 4288.25, 'median': 5282.5, 'q75': 6253.0, 'q90': 8472.0, 'max': 20679}`
- Preview flight index: `1`
- Preview flight shape: `(4723, 23)`

## Header Columns

before_after, date_diff, flight_length, label, hierarchy, fold, class, target_class, hclass, number_flights_before

## Fold x Label Counts

      before_after_0  before_after_1
fold                                
0               1167            1123
1               1123            1166
2               1162            1127
3               1192            1097
4               1200            1089

## Class-wise Flight Length Summary

              count      mean       std  min  median    max
before_after                                               
0              5844  5608.322  2220.502  605  5330.5  17632
1              5602  5422.832  2197.584  605  5212.0  20679

## Fold-wise Flight Length Summary

      count      mean       std  min  median    max
fold                                               
0      2290  5465.310  2147.040  605  5289.0  16698
1      2289  5505.979  2183.149  621  5283.0  17244
2      2289  5536.072  2273.368  605  5288.0  15578
3      2289  5517.135  2200.212  605  5258.0  20679
4      2289  5563.215  2250.456  605  5290.0  16024