import csv 
import numpy as np  
import pandas as pd 
from Tratamento_Dados import * 
from Reg_Algorithms import * 
import matplotlib.pyplot as plt 
from pandas_profiling import ProfileReport as pr 
import seaborn as sns 

Devolucoes = pd.read_csv('Datasets\GS_Projeto\devolucoes_csv.csv', encoding='latin', delimiter=';')

Entradas = pd.read_csv('Datasets\GS_Projeto\entradas_csv.csv', encoding='latin', delimiter=';')

Saidas = pd.read_csv('Datasets\GS_Projeto\saidas_csv.csv', encoding='latin', delimiter=';')

Saidas_B2B = pd.read_csv('Datasets\GS_Projeto\saidasB2B_csv.csv', encoding='latin', delimiter=';')

Prod_DF= pd.read_csv('Datasets/GS_Projeto/Prod_OVC_Filtered_csv.csv', encoding='latin', delimiter=';')

Prod_DF["Variant SKU"] = Prod_DF["Variant SKU"].str.slice(0,10)

Nm_Artigos_saidas = list(dict.fromkeys(Saidas['ReferênciaOrigem'].tolist()))
Nm_Artigos_saidas_b2b = list(dict.fromkeys(Saidas_B2B['ReferênciaOrigem'].tolist()))
Nm_Artigos_saidas_entr = list(dict.fromkeys(Entradas['ReferênciaOrigem'].tolist()))
Nm_Artigos_saidas_dev = list(dict.fromkeys(Devolucoes['ReferênciaOrigem'].tolist()))
Nm_Artigos_saidas_Prod = list(dict.fromkeys(Prod_DF["Variant SKU"].tolist()))

## Mudar nome Colunas Cada um dos Datasets

lista_colunas= list(Entradas.columns)
lista_colunas_prod= list(Prod_DF.columns)

['Semana', 'ReferênciaOrigem', 'Qnt']

Devolucoes = Devolucoes.rename(columns={'Qnt': 'Qnt_Devolucoes'})
Entradas = Entradas.rename(columns={'Qnt': 'Qnt_Entradas'})
Saidas = Saidas.rename(columns={'Qnt': 'Qnt_Saidas_B2C'})
Saidas_B2B = Saidas_B2B.rename(columns={'Qnt': 'Qnt_Saidas_B2B'})
Prod_DF = Prod_DF.rename(columns={'Variant SKU': 'ReferênciaOrigem'})

#resultado=(pd.merge(Saidas, Saidas_B2B, on=['ReferênciaOrigem']))

Resultado = Saidas.merge(Saidas_B2B, on=['Semana','ReferênciaOrigem'], how='outer') #.dropna(subset = ['descart', 'colcart'])
Resultado_ = Resultado.merge(Entradas, on=['Semana','ReferênciaOrigem'], how='outer')
DF_Final = Resultado_.merge(Devolucoes, on=['Semana','ReferênciaOrigem'], how='outer')

#saidas_concat = Saidas_B2B.loc[Saidas_B2B.year == 2016]

saidas_concat = Saidas.loc[Saidas.ReferênciaOrigem == "P143195000"]
saidas_b2b_concat = Saidas_B2B.loc[Saidas_B2B.ReferênciaOrigem == "P143195000"]
Resultado_concat = Resultado.loc[Resultado.ReferênciaOrigem == "P143195000"]

Contagem_Ids=len(DF_Final.dropna(subset = ['Qnt_Saidas_B2B']))

Prod_OVC_Inteiro=pd.read_csv('Datasets\GS_Analise_Dados\Produtos OVC_CSV.csv', encoding='latin', delimiter=';')
OVC_Filtered_Cols=['Handle', 'Title', 'Body (HTML)', 'Vendor', 'Standard Product Type', 'Custom Product Type', 'Tags', 'Published', 'Option1 Name', 'Option1 Value', 'Option2 Name', 'Option2 Value', 'Option3 Name', 'Option3 Value', 'Variant SKU', 'Variant Grams', 'Variant Inventory Tracker', 'Variant Inventory Qty', 'Variant Inventory Policy', 'Variant Fulfillment Service', 'Variant Price', 'Variant Compare At Price', 'Variant Requires Shipping', 'Variant Taxable', 'Variant Barcode', 'Image Src', 'Image Position','Status']
Prod_OVC_Inteiro=Prod_OVC_Inteiro[OVC_Filtered_Cols]
Prod_OVC_Inteiro = Prod_OVC_Inteiro.rename(columns={'Variant SKU': 'ReferênciaOrigem'})
Prod_OVC_Concat = DF_Final.merge(Prod_DF, on=['ReferênciaOrigem'], how='outer')

## Filtrar por valor -> confirmar existencia nos 2 datasets
Prod_OVC_Inteiro_analise = Prod_OVC_Inteiro.loc[Prod_OVC_Inteiro.ReferênciaOrigem == "P501164000"]

print(len(list(dict.fromkeys(DF_Final["ReferênciaOrigem"].tolist()))))

DF_Final=DF_Final.fillna(0)
DF_Final['Index'] = DF_Final.index

#DF_Final.to_excel('Datasets_Juntos.xlsx') ## Entradas, Saidas, Saidas_B2B, Devoluções

#####################

Prod_DF_Null=Prod_DF.dropna(subset = ['Title'])
Prod_OVC_NoNulls_Concat = DF_Final.merge(Prod_DF, on=['ReferênciaOrigem'], how='outer')

DF_all_concat = DF_Final.merge(Prod_DF_Null, on=['ReferênciaOrigem'], how='outer')
print(DF_all_concat['Title'].isnull().sum())
print(Prod_DF_Null['Title'].isnull().sum())
print(len(list(dict.fromkeys(DF_all_concat["ReferênciaOrigem"].tolist()))))
print(len(list(dict.fromkeys(Prod_DF_Null["Title"].tolist()))))
print(len(list(dict.fromkeys(Prod_DF_Null["ReferênciaOrigem"].tolist()))))
print(len(list(dict.fromkeys(DF_Final["ReferênciaOrigem"].tolist()))))

Lista_=[1,2,3,4,5,6,7,7,7,7,7,7,7]
print((list(dict.fromkeys(Lista_))))

lista_id_DF_Final=(list(dict.fromkeys(DF_Final["ReferênciaOrigem"].tolist())))
lista_id_Prod=(list(dict.fromkeys(Prod_DF_Null["ReferênciaOrigem"].tolist())))
lista_id_concat=(list(dict.fromkeys(DF_all_concat["ReferênciaOrigem"].tolist())))

diferenca=list(set(lista_id_DF_Final) - set(lista_id_Prod))
diferenca_inv=list(set(lista_id_Prod) - set(lista_id_DF_Final))
diferenca_total=list(set(lista_id_concat) - set(lista_id_DF_Final))

lista_colunas_DF_all_concat= list(DF_all_concat.columns)
lista_colunas_DF_all_concat=['Semana', 'ReferênciaOrigem', 'Qnt_Saidas_B2C', 'Qnt_Saidas_B2B', 'Qnt_Entradas', 'Qnt_Devolucoes', 'Index', 'Handle', 'Title', 'Body (HTML)', 'Vendor', 'Standard Product Type', 'Custom Product Type', 'Tags', 'Published', 'Option1 Name', 'Option1 Value', 'Option2 Name', 'Option2 Value', 'Option3 Name', 'Option3 Value', 'Variant Grams', 'Variant Inventory Tracker', 'Variant Inventory Qty', 'Variant Inventory Policy', 'Variant Fulfillment Service', 'Variant Price', 'Variant Compare At Price', 'Variant Requires Shipping', 'Variant Taxable', 'Variant Barcode', 'Image Src', 'Image Position', 'Status']

####

Dataset_Final=DF_all_concat.copy()

print(Dataset_Final['Qnt_Devolucoes'].isnull().sum())
print(Dataset_Final['Qnt_Saidas_B2C'].isnull().sum())
print(Dataset_Final['Qnt_Saidas_B2B'].isnull().sum())

Dataset_Final.dropna(subset = ["Qnt_Devolucoes"], inplace=True)  ## Eliminar linhas nulas

lista_id_Dataset_Final=(list(dict.fromkeys(Dataset_Final["ReferênciaOrigem"].tolist())))

diferenca_Dataset_Final=list(set(lista_id_Dataset_Final) - set(lista_id_DF_Final))
diferenca_Dataset_Final_2=list(set(lista_id_DF_Final) - set(lista_id_Dataset_Final))

Dataset_Final_copia=Dataset_Final.copy()

print(len(list(dict.fromkeys(Dataset_Final_copia["Title"].tolist()))))

Dataset_Final_copia.dropna(subset = ["Title"], inplace=True) 

##########################
### PERDA TOTAL DE ID's de ARTIGOS COM CONCETENAÇÃO
(len(list(dict.fromkeys(Dataset_Final["ReferênciaOrigem"].tolist()))))
(len(list(dict.fromkeys(Dataset_Final_copia["ReferênciaOrigem"].tolist()))     

#print("Perda total de Artigos com concatenação dos exceis extraidos ao longo do ano com os excel dos produtos referentes site é de ", 
#      (len(list(dict.fromkeys(Dataset_Final["ReferênciaOrigem"].tolist())))) - 
#       (len(list(dict.fromkeys(Dataset_Final_copia["ReferênciaOrigem"].tolist())))), 
#       "ficando com um conjunto de total de Artigos de",  (len(list(dict.fromkeys(Dataset_Final_copia["ReferênciaOrigem"].tolist()))
#      )))

Dataset_Final_copia.drop("Index", axis=1, inplace=True)
Dataset_Final_copia['Index'] = Dataset_Final_copia.index
#Dataset_Final_copia.to_excel('Junção_Total.xlsx') ## Entradas, Saidas, Saidas_B2B, Devoluções

Dataset_Final_copia.isnull().sum()

#### Corrigir Erros Title

DF_Edit=Dataset_Final_copia.copy()

for row in DF_Edit["Title"]:
    count=0
    if "?°" in row:
        row=row.replace("?°", "á") 
        count=count+1
    elif "?ß" in row: 
        row=row.replace("?ß", "ç")
        count=count+1
    elif "?£" in row:
        row=row.replace("?£", "ã")
        count=count+1
    elif "?©" in row:
        count=count+1
    print(count)

DF_Edit["Title"] = DF_Edit["Title"].str.replace("?°", "á",inplace=True)
DF_Edit["Title"] = DF_Edit["Title"].str.replace("?ß", "ç",inplace=True)
DF_Edit["Title"] = DF_Edit["Title"].str.replace("?£", "ã",inplace=True)
DF_Edit["Title"] = DF_Edit["Title"].str.replace("?©", "é",inplace=True)

DF_Edit["Title"] = DF_Edit["Title"].apply(lambda x: x.replace("?°", "á"))
DF_Edit["Title"] = DF_Edit["Title"].apply(lambda x: x.replace("?ß", "ç"))
DF_Edit["Title"] = DF_Edit["Title"].apply(lambda x: x.replace("?£", "ã"))
DF_Edit["Title"] = DF_Edit["Title"].apply(lambda x: x.replace("?©", "é"))

list(Prod_DF["ReferênciaOrigem"].values)

DF_Edit.drop("Index", axis=1, inplace=True)
DF_Edit['Index'] = Dataset_Final_copia.index
DF_Edit.to_excel('Junção_Final.xlsx') ## Entradas, Saidas, Saidas_B2B, Devoluções

################## EDIÇÃO DATASET #####################

Juncao_DF = pd.read_csv('Datasets\GS_Projeto\Junção_Final_csv.csv', encoding='latin', delimiter=';')

Gs_df=Juncao_DF.copy()
Gs_df=Gs_df.drop(columns=['Unnamed: 0'],axis=1)
Gs_df_columns= list(Gs_df.columns)

Gs_df['Data'] = Gs_df.loc[:, 'Semana']

from datetime import date, timedelta,datetime  

def allmondays(year):
   d = date(year, 1, 1)                    # January 1st
   d += timedelta(days = 6 - d.weekday())  # First Sunday
   while d.year == year:
      yield d
      d += timedelta(days = 7)

Lista_dias=[]
for d in allmondays(2021):
    d=d + timedelta(days=1)
    Lista_dias.append(d)
    print(d)

Lista_semanas=(list(dict.fromkeys(Gs_df["Semana"].tolist())))
Lista_semanas.sort()

Lista_dias=Lista_dias[:48]

Gs_df["Data"]=Gs_df["Data"].replace(Lista_semanas, Lista_dias) ### Separar Lista de Valores de Coluna por Outra Lista

Gs_df['Estação'] = Gs_df.loc[:, 'Semana']

list_inverno=list(range(1,13))
list_Primavera=list(range(13,26))
list_Verão=list(range(26,39))
list_Outono=list(range(39,50))

for valores in Lista_semanas:
    if valores in list_inverno:
        #Gs_df['Estação'].loc[(Gs_df['Estação'] < 13)] = "Inverno" ## Substituir Valores baseado em condições
        Gs_df.loc[Gs_df["Estação"] == valores, "Estação"] = "Inverno"
    elif valores in list_Primavera:
        #Gs_df['Estação'].loc[(Gs_df['Estação'] < 26)] = "Primavera"
        Gs_df.loc[Gs_df["Estação"] == valores, "Estação"] = "Primavera"
    elif valores in list_Verão:
        #Gs_df['Estação'].loc[(Gs_df['Estação'] < 39)] = "Verão"
        Gs_df.loc[Gs_df["Estação"] == valores, "Estação"] = "Verão"
    elif valores in list_Outono:
        Gs_df.loc[Gs_df["Estação"] == valores, "Estação"] = "Outono"
       #Gs_df['Estação'].loc[(Gs_df['Estação'] < 50)] = "Outono"     
       
Gs_df.dtypes

# Gs_df['Data']=pd.to_object(Gs_df['Data']) # Converter Coluna em Data

Gs_df['Data']=Gs_df['Data'].astype(str) ## Converter Coluna em String 


### Criar Coluna mês ###
Gs_df['Mês'] = Gs_df['Data'].apply(lambda x: str(x)[5:7])

lista_num_meses=(list(dict.fromkeys(Gs_df["Mês"].tolist())))
lista_num_meses=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
lista_Meses=["Janeiro","Feveiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro"]

Gs_df["Mês"]=Gs_df["Mês"].replace(lista_num_meses, lista_Meses)

Gs_df.dropna(subset = ["Index"], inplace=True) 

Gs_df["Body (HTML)"]= Gs_df["Body (HTML)"].str.replace("?°", "á",inplace=True)

Gs_df["Body (HTML)"] = Gs_df["Body (HTML)"].apply(lambda x: x.replace("?°", "á"))
Gs_df["Body (HTML)"] = Gs_df["Body (HTML)"].apply(lambda x: x.replace("?ß", "ç"))
Gs_df["Body (HTML)"] = Gs_df["Body (HTML)"].apply(lambda x: x.replace("?£", "ã"))
Gs_df["Body (HTML)"] = Gs_df["Body (HTML)"].apply(lambda x: x.replace("?©", "é"))

Gs_df.dtypes

Gs_df.to_excel('Gs_02-02.xlsx')

Gs_df["Body (HTML)"].head()

"""
################################################################################################################
################################################################################################################
##### POWER BI ######
"""

##1) PBI_Vendas_Totais
Df_1 = pd.read_csv('Datasets\PowerBi\PBI_Vendas_Totais_1.csv', encoding='latin', delimiter=',')
Data=Df.copy()
Colunas=list(Data.columns)


Data=Data.dropna(subset = ['forecastValue'])

print(Data['ï»¿sales'], 
      Data['forecastValue'])
metricas_1=pd.DataFrame(metricas_regression(Data['ï»¿sales'], Data['forecastValue']),index=[0])
previsao=[len(Data['forecastValue'])]
metricas_1['Previsão Dias'] = previsao
print(len(Data['ï»¿sales']))


print(Data['forecastValue'][0:])
def forecast_powerbi(Data):
    Data=Data.dropna(subset = ['forecastValue'])
    metricas=pd.DataFrame(metricas_regression(Data['ï»¿sales'], Data['forecastValue']),index=[0])
    previsao=[len(Data['forecastValue'])]
    metricas['Forecast lenght'] = previsao
    return metricas

metricas=forecast_powerbi(Data)

##2) PBI PBI_P144205001_90dias

Df_2 = pd.read_csv('Datasets\PowerBi\PBI_P144205001_90dias_Atraso_7dias_Forecast_2.csv', encoding='latin', delimiter=',')
Data_2=Df_2.copy()
Colunas_2=list(Data_2.columns)

Data_2=Data_2.dropna(subset = ['forecastValue'])

print(Data_2['ï»¿sales'], 
      Data_2['forecastValue'])

metricas_2=pd.DataFrame(metricas_regression(Data_2['ï»¿sales'], Data_2['forecastValue']),index=[0])

##3) PBI PBI_P144205001_90dias_Atraso_7dias_Forecast

Df_3 = pd.read_csv('Datasets\PowerBi\PBI_P142458018_60dias_Atraso_14dias_Forecast_3.csv', encoding='latin', delimiter=',')
Data_3=Df_3.copy()
Colunas_3=list(Data_3.columns)

Data_3=Data_3.dropna(subset = ['forecastValue'])

print(Data_3['ï»¿sales'], 
      Data_3['forecastValue'])

metricas_3=pd.DataFrame(metricas_regression(Data_3['ï»¿sales'], Data_3['forecastValue']),index=[0])

##### POWER BI - Analise de Algoritmo Forecast ######

Df_1 = pd.read_csv('Datasets\PowerBi_\PBI_Vendas_Totais_1.csv', encoding='latin', delimiter=',')
Df_2 = pd.read_csv('Datasets\PowerBi_\PBI_P144205001_90dias_Atraso_7dias_Forecast_2.csv', encoding='latin', delimiter=',')
Df_3 = pd.read_csv('Datasets\PowerBi_\PBI_P142458018_60dias_Atraso_14dias_Forecast_3.csv', encoding='latin', delimiter=',')
Df_4 = pd.read_csv('Datasets\PowerBi_\PBI_P500431000_120dias_Atraso_14dias_Forecast_4.csv', encoding='latin', delimiter=',')
Df_5 = pd.read_csv('Datasets\PowerBi_\PBI_P144539003_21dias_Atraso_7dias_Forecast_5.csv', encoding='latin', delimiter=',')
Df_6 = pd.read_csv('Datasets\PowerBi_\PBI_P974715002_14dias_Atraso_7dias_Forecast_6.csv', encoding='latin', delimiter=',')

def forecast_powerbi(Data):
    Data=Data.dropna(subset = ['forecastValue'])
    metricas=pd.DataFrame(metricas_regression(Data['ï»¿sales'], Data['forecastValue']),index=[0])
    previsao=[len(Data['forecastValue'])]
    metricas['Forecast lenght'] = previsao
    return metricas

metricas_1=forecast_powerbi(Df_1)
metricas_2=forecast_powerbi(Df_2)
metricas_3=forecast_powerbi(Df_3)
metricas_4=forecast_powerbi(Df_4)
metricas_5=forecast_powerbi(Df_5)
metricas_6=forecast_powerbi(Df_6)
lista_dfs=[metricas_1,metricas_2,metricas_3,metricas_4,metricas_5,metricas_6]

def concat_dfs(Lista_Metricas:list):

    Modelos_Treino=Lista_Metricas.copy()
    Modelos_Treino_DF=[]

    for elementos in Modelos_Treino:
        Modelos_Treino_DF.append(elementos)

    Agrup_Modelos_Lider=pd.concat(Modelos_Treino_DF) #### Implementar Agrup_Modelos_Lider sorted !!!!!!!!!!!
    Agrup_Modelos_Lider=Agrup_Modelos_Lider.reset_index()
    Agrup_Modelos_Lider.drop(Agrup_Modelos_Lider.columns[0], axis=1, inplace=True)
    
    return Agrup_Modelos_Lider

metricas_totais=concat_dfs(lista_dfs)


#########################################################################################################################



Dados_Vendas = pd.read_csv('Datasets\GS_Projeto\dados_vendas_csv.csv', encoding='latin', delimiter=';')

Dados_Vendas['Nova_Data'] = Dados_Vendas.Data.str[0:10]
Dados_Vendas["Nova_Data"]=pd.to_datetime(Dados_Vendas['Nova_Data'])
Dados_Vendas_=Dados_Vendas.groupby(['Nova_Data'])['Qnt'].sum().reset_index()
Dados_Vendas=Dados_Vendas.drop("Nova_Data",axis=1)

Dados_Vendas["Data"]=pd.to_datetime(Dados_Vendas['Data'])
Dados_Vendas["Data"] = Dados_Vendas["Data"].astype(object)



Dados_Vendas.dtypes

#Dados_Vendas_=Dados_Vendas.groupby(['Data'])['Qnt'].sum().reset_index() ## Agrupar Produtos por Quantidade


Dados_Vendas_=Dados_Vendas.groupby(['Data'])['Qnt'].agg('sum')

























