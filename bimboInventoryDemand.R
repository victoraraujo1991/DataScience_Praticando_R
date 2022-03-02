# Definição do diretório de trabalho
setwd("D:/Documentos/Cursos/BigDataRAzure/TrabalhoFinal-Regressao")
getwd()

## Título: Projeto com Feedback 2 - Prevendo demanda de estoque com Base em Vendas

## Para a construção desse projeto, será utilizado o dataset disponibilizado no Kaggle, pelo grupo Bimbo:
# https://www.kaggle.com/c/grupo-bimbo-inventory-demand

## O Objetivo deste projeto é desenvolver um modelo para prever com precisão a demanda de estoque com base
# nos dados históricos de vendas, para o grupo Bimbo. Atualmente, os cálculos diários de estoque são
# realizados por funcionários de vendas de entregas diretas, que devem, sozinhos, prever a necessidade
# do estoque dos produtos e demanda com base em suas experiências pessoais em cada loja.

## Primeiramente, necessário fazer a leitura dos arquivos de treino
# Vamos utilizar o tidyverse que contém vários pacotes úteis para o tratamento e visualização dos dados
# install.packages("tidyverse")
library("tidyverse")
library("lubridate")
library("chron")
library("caTools")
library("caret")

#### (uncomment quando for utilizar o dataset original)
## Leitura do dataset de treino Original 
# df_Treino = read.csv("train.csv")


#### (uncomment quando for utilizar o dataset parcial)
## Devido as limitações do hardware utilizado, vamos particionar o dataset em um número menor
# e após a criação do modelo, será substituído pelo dataset original com todos os dados
# amostra <- sample_n(df_Treino, 2000000)

# Gravar o arquivo para não precisar ficar particionando o dataset original toda vez que retornar ao código
# write.csv(amostra, "D:/Documentos/Cursos/BigDataRAzure/TrabalhoFinal-Regressao/short_train.csv", row.names = FALSE)

# Leitura do dataset de treino
df_Treino = read.csv("short_train.csv")

# Breve avaliação das variáveis do dataset
head(df_Treino)
str(df_Treino)
summary(df_Treino)

## Uma breve análise exploratória dos dados para entender um pouco mais sobre o dataset
# Existem dados NA no dataframe?
sum(is.na(df_Treino))

# Qual produto mais vendido?
dfTreino_graf1 <- count(df_Treino, Producto_ID)
dfTreino_graf1 <- data.frame(dfTreino_graf1)
colnames(dfTreino_graf1) <- c('Produto','Qtd')
dfTreino_graf1$Produto <- as.factor(dfTreino_graf1$Produto)
dfTreino_graf1 <- arrange(dfTreino_graf1, desc(Qtd))
dfTreino_graf1 <- head(dfTreino_graf1, 10)

dfTreino_graf1 %>%
  ggplot(aes(y = fct_rev(fct_reorder(Produto, desc(Qtd))),
             Qtd))+
  geom_col() +
  geom_text(aes(label = Qtd),
            hjust = -0.3,
            vjust = 0.5,
            size = 3.5) +
  labs(title = "TOP 10 mais vendidos", 
       subtitle = "Quantidade de produtos vendidos no período analisado",
       x = "Quantidade de produtos vendidos",
       y = "Producto_ID")

# Qual cliente mais comprou no período analisado?
dfTreino_graf2 <- count(df_Treino, Cliente_ID)
dfTreino_graf2 <- data.frame(dfTreino_graf2)
colnames(dfTreino_graf2) <- c('Cliente','Qtd')
dfTreino_graf2$Cliente <- as.factor(dfTreino_graf2$Cliente)
dfTreino_graf2 <- arrange(dfTreino_graf2, desc(Qtd))
dfTreino_graf2 <- head(dfTreino_graf2, 10)

dfTreino_graf2 %>%
  ggplot(aes(y = fct_rev(fct_reorder(Cliente, desc(Qtd))),
             Qtd))+
  geom_col() +
  geom_text(aes(label = Qtd),
            hjust = -0.3,
            vjust = 0.5,
            size = 3.5) +
  labs(title = "TOP 10 clientes", 
       subtitle = "Quantidade de compras por clientes no período analisado",
       x = "Quantidade de compras por clientes",
       y = "Cliente")

# Qual faturamento semanal?
ggplot(data = df_Treino, aes(x = Semana, y = Venta_hoy)) +
  geom_col() +
  ggtitle("Faturamento semanal da empresa") +
  labs(x="Faturamento", y="Semana") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Vamos fazer uma análise da relação das variáveis
modeloVar <- train(Demanda_uni_equil ~ ., data = df_Treino, method = "lm")
varImp(modeloVar)

## Treinando o modelo
# Não podemos utilizar as variáveis com maior relação com a variável demanda, pois elas não estão
# presentes no dataset de teste
firstModel <- lm(Demanda_uni_equil ~ Ruta_SAK + Producto_ID + Canal_ID + Cliente_ID, 
                 data = df_Treino)

# Analisar o modelo criado
summary(firstModel)

# Leitura dos dados de testes do Kaggle para realizar a atividade:
df_Teste = read.csv("test.csv")

# Aplicar sobre os dados de teste
previsao <- predict(firstModel, df_Teste)

# Com o modelo atual, conseguimos atingir um R² de 2,2%. Um valor pífio. Precisamos trabalhar esse modelo.

## Iniciar tratativa dos dados
# Converter os tipos de dados para tipos mais apropriados
df_Treino$Semana <- as.numeric(df_Treino$Semana)
df_Treino$Agencia_ID <- as.numeric(df_Treino$Agencia_ID)
df_Treino$Ruta_SAK <- as.numeric(df_Treino$Ruta_SAK)
df_Treino$Canal_ID <- as.numeric(df_Treino$Canal_ID)
df_Treino$Cliente_ID <- as.numeric(df_Treino$Cliente_ID)
df_Treino$Producto_ID <- as.numeric(df_Treino$Producto_ID)

# Verificar dados depois dos ajustes
str(df_Treino)
summary(df_Treino)

# Avaliando as variáveis int e num, podemos identificar que existem valores muito longes da média e mediana
# Estes valores são outliers e precisam ser retirados da análise

## Retirar do df alguns outliers da coluna Venta_uni_hoy
# Identificamos os outliers através de uma análise das variância e desvio padrão
sd(df_Treino$Venta_uni_hoy)
var(df_Treino$Venta_uni_hoy)
# Deletamos os valores da tabela acima de média + desvio padrão
df_Treino2 <- subset(df_Treino, Venta_uni_hoy < 30 )

## Retirar do df alguns outliers da coluna Venta_hoy
# Identificamos os outliers através de uma análise das variância e desvio padrão
summary(df_Treino2)
sd(df_Treino2$Venta_hoy)
var(df_Treino2$Venta_hoy)
# Deletamos os valores da tabela acima de média + desvio padrão
df_Treino2 <- subset(df_Treino2, Venta_hoy < 100 )

## Retirar do df alguns outliers da coluna Dev_uni_proxima
# Identificamos os outliers através de uma análise das variância e desvio padrão
summary(df_Treino2)
sd(df_Treino2$Dev_uni_proxima)
var(df_Treino2$Dev_uni_proxima)
# Deletamos os valores da tabela acima de média + desvio padrão
df_Treino2 <- subset(df_Treino2, Dev_uni_proxima < 5 )

## Retirar do df alguns outliers da coluna Dev_proxima
# Identificamos os outliers através de uma análise das variância e desvio padrão
summary(df_Treino2)
sd(df_Treino2$Dev_proxima)
var(df_Treino2$Dev_proxima)
# Deletamos os valores da tabela acima de média + desvio padrão
df_Treino2 <- subset(df_Treino2, Dev_proxima < 4 )

## Retirar do df alguns outliers da coluna Demanda_uni_equil
# Identificamos os outliers através de uma análise das variância e desvio padrão
summary(df_Treino2)
sd(df_Treino2$Demanda_uni_equil)
var(df_Treino2$Demanda_uni_equil)
# Deletamos os valores da tabela acima de média + desvio padrão
df_Treino2 <- subset(df_Treino2, Demanda_uni_equil < 8 )

# Verificar dados depois dos ajustes
str(df_Treino2)
summary(df_Treino2)

## Vamos colocar algumas médias de valores no dataframe de testes
# Agregando por cliente, produto e ruta_sak com o valor da mediana de Venta_uni_hoy para essas variáveis
newColuna <- aggregate(df_Treino2$Venta_uni_hoy, 
                       list(df_Treino2$Cliente_ID), 
                       FUN=median)
# Altera o nome das colunas do dataframe
colnames(newColuna) <- c('Cliente_ID', 'Venta_uni_hoy')

# Agregando por cliente, produto e ruta_sak com o valor da mediana de Dev_uni_proxima para essas variáveis
newColuna2 <- aggregate(df_Treino2$Dev_uni_proxima, 
                       list(df_Treino2$Cliente_ID), 
                       FUN=median)
# Altera o nome das colunas do dataframe
colnames(newColuna2) <- c('Cliente_ID', 'Dev_uni_proxima')

# Vamos criar um novo modelo, apenas com as variáveis mais relevantes do dataset

## Treinando a segunda versão do modelo
secondModel <- lm(Demanda_uni_equil ~ Venta_uni_hoy + Dev_uni_proxima, 
                 data = df_Treino2)

# Analisar o modelo criado
summary(secondModel)

# As novas colunas criadas, serão inseriadas no dataframe de testes
df_TesteFinal <- full_join(df_Teste, newColuna, 
                       by.x = c("Cliente_ID"),
                       by.y = c("Cliente_ID"),
                       all.x = TRUE,
                       all.y = TRUE)

# As novas colunas criadas, serão inseriadas no dataframe de testes
df_TesteFinal <- merge(df_TesteFinal, newColuna2, 
                       by.x = c("Cliente_ID"),
                       by.y = c("Cliente_ID"),
                       all.x = TRUE,
                       all.y = TRUE)

df_TesteFinal <- arrange(df_TesteFinal, id)
                       
# Aplicar sobre os dados de teste
previsao2 <- predict(secondModel, df_TesteFinal)
sum(is.na(previsao2))


# Criar o arquivo de teste para o Kaggle
df_TesteReal_Kaggle <- mutate(df_TesteFinal,
                              previsao2)

df_TesteReal_Kaggle <- select(df_TesteReal_Kaggle, id, previsao2)

# Altera o nome das colunas do dataframe para teste no Kaggle
colnames(df_TesteReal_Kaggle) <- c('id', 'Demanda_uni_equil')
head(df_TesteReal_Kaggle)

write.csv(df_TesteReal_Kaggle, "D:/Documentos/Cursos/BigDataRAzure/TrabalhoFinal-Regressao/submissionKaggle.csv", row.names = FALSE)

# Por se tratar de um exercício e ter um curso ainda bem longo pela frente, vou finalizar por aqui
