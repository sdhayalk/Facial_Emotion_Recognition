# referred from: https://github.com/npinto/fer2013/blob/master/convert_fer2013.py

import csv

csvr = csv.reader(open('fer2013.csv'))
header = next(csvr)
rows = [row for row in csvr]

trn = [row[:-1] for row in rows if row[-1] == 'Training']
csv.writer(open('training.csv', 'w+')).writerows([header[:-1]] + trn)
print(len(trn))

tst = [row[:-1] for row in rows if row[-1] == 'PublicTest']
csv.writer(open('test.csv', 'w+')).writerows([header[:-1]] + tst)
print(len(tst))

tst2 = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
csv.writer(open('testprivate.csv', 'w+')).writerows([header[:-1]] + tst2)
print(len(tst2))