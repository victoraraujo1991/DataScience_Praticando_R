# Definição do diretório de trabalho
setwd("D:/Documentos/Cursos/BigDataRAzure/TrabalhoFinal-Classificacao")
getwd()

# Nesta atividade, a missão é criar um algoritmo para prever se um clique é fraudulento ou não.
# Nesta atividade será construído um modelo de aprendizagem de máquina para prever se um usuário fará
# o download de um aplicativo após clicar em um anúncio de aplicativo para dispositivos móveis

# Para a construção desse projeto, será utilizado o dataset disponibilizado no Kaggle, 
# pela empresa TalkingData
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

# Este dataset cobre um período de 4 dias, onde são fornecidos informações de aprox. 200 milhões de cliques

# Primeiramente, necessário fazer a leitura dos arquivos de treino
# Vamos utilizar o tidyverse que contém vários pacotes úteis para o tratamento e visualização dos dados
# install.packages("tidyverse")
library("tidyverse")
library("lubridate")
library("chron")
library("caTools")
library("randomForest")
library("caret")

## Como os dados são muito pesados, vou montar as análises exploratórias com apenas o arquivo train_sample
df_Treino = read.csv("train_sample.csv")

# Com os dados carregados, primeiramente necessário realizar uma análise exploratória
# Para entender o tipo de dados que serão trabalhos e conseguir alguns insights imediatos

head(df_Treino)
str(df_Treino)
summary(df_Treino)

# Converter os tipos de dados para tipos mais apropriados
# Converte os campos device, os para numeric e is_attributed para factor
df_Treino$device <- as.numeric(df_Treino$device)
df_Treino$os <- as.numeric(df_Treino$os)
df_Treino$is_attributed <- as.factor(df_Treino$is_attributed)

# Converte os campos clicktime, attributed_time para POSIXct
df_Treino$click_time <- parse_date_time(df_Treino$click_time, orders="ymd HMS")
df_Treino$attributed_time <- parse_date_time(df_Treino$attributed_time, orders="ymd HMS")

# Criar uma coluna com o dia e outra com a hora para as duas variáveis date Time
dfNewColumnClickTime_time <- format(as.POSIXct(df_Treino$click_time), format="%H:%M:%S")
dfNewColumnClickTime_date <- format(as.POSIXct(df_Treino$click_time), format="%Y-%m-%d")
dfNewColumnAttributed_time <- format(as.POSIXct(df_Treino$attributed_time), format="%H:%M:%S")
dfNewColumnAttributed_date <- format(as.POSIXct(df_Treino$attributed_time), format="%Y-%m-%d")

# Criar nova coluna informando se o período do dia é manhã, tarde, noite ou madrugada
dfNewColumnClickTime_Hora <- format(as.POSIXct(df_Treino$click_time), format="%H")

dayOrNight <- function(x){
  if (x >= 0 & x <= 5)
    return ("MADRUGADA")
  else if (x >= 6 & x <= 12)
    return ("MANHA")
  else if (x >= 13 & x <= 18)
    return ("TARDE")
  else if (x >= 19 & x <= 23)
    return ("NOITE")
}

dfNewColumnShift <- sapply(as.integer(dfNewColumnClickTime_Hora), dayOrNight)
dfNewColumnShift <- data.frame(dfNewColumnShift)

# Criar nova coluna informando o dia da semana (segunda, terça, quarta.. etc.)
diaSemana <- wday(dfNewColumnClickTime_date, label = TRUE)
diaSemana <- data.frame(diaSemana)

# Juntar todas as colunas em um novo data frame
dfTreino_New <- mutate(df_Treino,
                       dfNewColumnClickTime_time,
                       dfNewColumnClickTime_date,
                       dfNewColumnAttributed_time,
                       dfNewColumnAttributed_date,
                       dfNewColumnShift,
                       diaSemana)

# Altera o nome das colunas do dataframe
colnames(dfTreino_New) <- c('ip', 'app', 'device', 'os', 'chanel', 'clickTimeDT', 'attributedTimeDT',
                            'isAttributed', 'clickTime', 'clickDate', 'AttributedTime', 'AttributedDate',
                            'shift', 'weekday')

# Converte os campos clicktime, attributed_time para tempo
dfTreino_New$clickTime <- hms(dfTreino_New$clickTime)
dfTreino_New$clickDate <- as.Date(dfTreino_New$clickDate)
dfTreino_New$AttributedTime <- hms(dfTreino_New$AttributedTime)
dfTreino_New$AttributedDate <- as.Date(dfTreino_New$AttributedDate)

# Avaliando como ficaram os dados depois das correções
head(dfTreino_New)
str(dfTreino_New)

## Insights analisados

# Qual período do dia em que ocorrem mais cliques? 
hist(dfTreino_New$clickTime)

# Gráfico de barras contendo 4 colunas (manhã, tarde, noite, madrugada), indicando em qual período do dia 
# mais ocorrer downloads de apps
ggplot(data = dfTreino_New, aes(x = shift)) +
  geom_bar()+
  ggtitle("Quantidade de Cliques por turno") +
  labs(x="Turno", y="Quantidade de Cliques") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Qual a relação entre cliques apps baixados X cliques sem baixar apps
table(dfTreino_New['isAttributed'])

# Criar modelo de análise para relação das variáveis
modeloVar <- randomForest(isAttributed ~ ip + app + device + os + chanel + clickTime + clickDate + shift + weekday,
                       data = dfTreino_New, ntree = 100, nodesize = 10, importance = T)

# Avaliar a relação das colunas para predição da variável
# Nesta avaliação vamos identificar que as variáveis app, ip, chanel e device possui uma maior relação com 
# baixar ou não os app. Desta forma, nosso modelo será criado utilizando essas variáveis.
varImpPlot(modeloVar)

# Split dos dados em 70% treino e 30% teste
split = sample.split(dfTreino_New$isAttributed, SplitRatio = 0.70)

# Datasets de treino e de teste
dados_treino = subset(dfTreino_New, split == TRUE)
dados_teste = subset(dfTreino_New, split == FALSE)

## Treinando o modelo
modeloFinal <- randomForest(isAttributed ~ ip + app + device + os + chanel + clickTime + shift,
                          data = dados_treino, ntree = 100, nodesize = 10, importance = T)

# Analisar o modelo criado
modeloFinal

# Aplicar sobre os dados de teste
previsao <- predict(modeloFinal, dados_teste)
previsao

# Criando uma confusion matrix
table(dados_teste$isAttributed, previsao)
confusionMatrix(table(dados_teste$isAttributed, previsao), positive = "1")

## Aparentemente o modelo atingiu 99,8% de precisão, mas isso pode ser um dado enganoso
# pois temos uma grande quantidade de dados 0 e uma quantidade muito pequena de dados 1
# Através da Confusion Matrix, verificamos com isso a seguinte situação:
# - Das 29932 linhas do DF onde o usuário não baixou o app, o modelo errou em apenas 0,006%
# - Das 68 linhas do DF onde o usuário realmente clicou e baixo o app, o modelo errou em 7% dos casos,
# ou seja, uma precisão de 93%.

## Vamos balancear o dataset e fazer novamente os testes:
# Primeiro, será criado um dataframe apenas com isAttributed == 1
dfTreino_only1 <- dfTreino_New %>% 
  filter(isAttributed == 1)

# Vamos avaliar quantas amostras foram filtradas
dfTreino_only1_lenght <- as.integer(count(dfTreino_only1))

# E criamos um novo dataframe com a variável isAttributed == 0 com o mesmo tamanho do df anterior
dfTreino_only0 <- dfTreino_New %>% 
  filter(isAttributed == 0) %>% 
  sample_n(size = dfTreino_only1_lenght)

# Unimos e criamos um novo data frame com o mesmo número de isAttributed == 1 e 0
dfTreino_otimizado1 <- bind_rows(dfTreino_only1, dfTreino_only0)

## Fazemos todos os testes novamente
# Criar modelo de análise para relação das variáveis
modeloVar2 <- randomForest(isAttributed ~ ip + app + device + os + chanel + clickTime + clickDate + shift + weekday,
                          data = dfTreino_otimizado1, ntree = 100, nodesize = 10, importance = T)

# Avaliar a relação das colunas para predição da variável
# Nesta avaliação vamos identificar que as variáveis app, ip, chanel e device possui uma maior relação com 
# baixar ou não os app. Desta forma, nosso modelo será criado utilizando essas variáveis.
varImpPlot(modeloVar2)

# Split dos dados em 70% treino e 30% teste
set.seed(256)
split = sample.split(dfTreino_otimizado1$isAttributed, SplitRatio = 0.70)

# Datasets de treino e de teste
dados_treino2 = subset(dfTreino_otimizado1, split == TRUE)
dados_teste2 = subset(dfTreino_otimizado1, split == FALSE)

## Treinando o modelo
modeloFinal2 <- randomForest(isAttributed ~ app + ip + device + os + chanel + clickTime,
                            data = dados_treino2, ntree = 100, importance = T)

# Analisar o modelo criado
modeloFinal2

# Aplicar sobre os dados de teste
previsao2 <- predict(modeloFinal2, dados_teste2)
previsao2

# Criando uma confusion matrix
table(dados_teste2$isAttributed, previsao2)
confusionMatrix(table(dados_teste2$isAttributed, previsao2), positive = "1")

# O modelo atingiu 92,6% de precisão com os dados balanceados
# Por se tratar de um exercício e ter um curso ainda bem longo pela frente, vou finalizar por aqui
