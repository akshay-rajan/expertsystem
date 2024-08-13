import csv

with open('./House_Rent_Dataset.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)