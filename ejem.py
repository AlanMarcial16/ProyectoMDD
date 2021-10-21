import pandas as pd
pdf_txt = 'data.txt'
openfile = open(pdf_txt, 'r')
for line in openfile:
  print (line)
df = pd.read_csv("data.txt", sep = ', ', header = None, thousands = ',')
df.head()
