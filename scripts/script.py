import pandas as pd

my_file = "train.csv"
df = pd.read_csv(my_file)

#conta quantidade de turnos na coluna "Context"
#Sabendo que "__eot__" significa um turno
#Em todo o corpus
df['count'] = df['Context'].str.count("__eot__")

#divide por quantidade
um = df.loc[df['count'] == 1] #pega somente as linhas com um turno
um.drop('count', axis=1, inplace=True) #remove a coluna count
print("Um turno:", len(um.index)) #imprime quantidade de exemplos encontrados

dois = df.loc[df['count'] == 2]
dois.drop('count', axis=1, inplace=True)
print("Dois turnos:", len(dois.index))

tres = df.loc[df['count'] == 3]
tres.drop('count', axis=1, inplace=True)
print("Três turnos:", len(tres.index))

quatro = df.loc[df['count'] == 4]
quatro.drop('count', axis=1, inplace=True)
print("Quatro turnos:", len(quatro.index))

#grava em arquivos csv, removendo a coluna de index pois é desnecessária
um.to_csv("um.csv", sep=',', index=False)
dois.to_csv("dois.csv", sep=',', index=False)
tres.to_csv("tres.csv", sep=',', index=False)
quatro.to_csv("quatro.csv", sep=',', index=False)