---
title: "TitanicDocumentation"
author: "Victor de Carvalho Martins Araujo"
date: '`r Sys.Date()`'
output:
  html_document:
    number_sections: true
    toc: true
---

Titanic Machine Learning from Disaster
Kaggle Project

Vamos prever uma classificação - sobreviventes e não sobreviventes

Utilizaremos a biblioteca tidyverse que contém várias outras bibliotecas úteis para nosso processo de data wrangling
```{r}
library("tidyverse")
```

Comecamos carregando o dataset de treino e também o de testes
```{r}
dados_treino <- read.csv('../input/titanic/train.csv')
dados_teste <- read.csv('../input/titanic/test.csv')
```

Vamos avaliar a estrutura do dataset, suas variáveis, tipo das variáveis e as primeiras amostras:
```{r}
str(dados_treino)
```

Podemos avaliar que temos nesse dataset de treino, 891 observações com 12 variáveis.

Para entendermos melhor todo esse conjunto de dados, vamos projetar alguns gráficos sobre todo esse conjunto de dados fornecidos.

```{r}
ggplot(dados_treino,aes(x=Survived)) + 
  geom_bar(colour="black", fill="#6383b8") + 
  ggtitle("Numbers of Titanic Survivors (0 = Not survived / 1 = Survived)") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```


```{r}
ggplot(dados_treino,aes(x=Survived)) + 
  geom_bar(colour="black", fill="#6383b8") + 
  ggtitle("Numbers of Titanic Survivors (0 = Not survived / 1 = Survived)") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```

```{r}
ggplot(dados_treino,aes(Pclass)) + 
  geom_bar(aes(fill = factor(Pclass)), alpha = 0.7) +
  ggtitle("Number of Passengers by Class") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```


```{r}
ggplot(dados_treino,aes(Sex)) + 
  geom_bar(aes(fill = factor(Sex)), alpha = 0.7) + 
  ggtitle("Sex of Passengers") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```


```{r}
ggplot(dados_treino,aes(Age)) + 
  geom_histogram(fill = '#6383b8', bins = 10, alpha = 0.7) +
  ggtitle("Histogram of Passengers Age's")
```


```{r}
ggplot(dados_treino,aes(SibSp)) +
  labs(title = "Number of Siblings or Spouses", 
       subtitle = "Showing the number of members by the number of siblings or spouses aboard the Titanic?",
       x = "Siblings or Spouses by person") +
  geom_bar(fill = 'red', alpha = 0.7) +
  scale_x_continuous(breaks = c(0,1,2,3,4,5,6,7,8)) +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```


```{r}
ggplot(dados_treino,aes(Parch)) + 
  labs(title = "Number of Parents or Children", 
       subtitle = "Showing the number of members by the number of parents or childers aboard the Titanic?",
       x = "Parents or Children by person") +
  geom_bar(fill = 'red', alpha = 0.7) +
  scale_x_continuous(breaks = c(0,1,2,3,4,5,6)) +
  geom_text(stat='count', aes(label=..count..), vjust=-0.2) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
```

```{r}
  ggplot(dados_treino,aes(Fare)) + 
    geom_histogram(fill = '#6383b8', bins = 20, alpha = 0.7) +
    ggtitle("Histogram of Passengers Fare")
```


Vamos avaliar a quantidade de dados NA (Not Available):
```{r}
print("Quantidade de dados NA:")
sum(is.na(dados_treino))
```

Ok, temos 177 valores NA. Vamos localizá-los em nosso dataset. Como vimos na apresentação das estruturas dos dados nos comandos acima, a variável Age possui um valor NA logo no início. Vamos avaliar quantos valores NA possui a variável "Age" de nosso dataset:

```{r}
print("Quantidade de dados NA na variável Age:")
sum(is.na(dados_treino$Age))
```

Ou seja: todos nossos dados NA encontram-se na mesma variável do dataset.

Identificado variáveis que não são uteis para a tratativa dos dados

As variáveis PassengerId e Ticket não são úteis para nosso algorítmo de classificação, pois uma é referente a um número de identificação criado para gerar o dataset e o outro é o número do ticket.
```{r}
dados_treino2 <- transmute(dados_treino,
                           Survived, Pclass, Sex, Age, SibSp, Parch)
```

Separar os dados de treino em uma tabela com valores NA e outra sem valores Na na coluna Age:
```{r}
dados_treino_com_NA <- filter(dados_treino2, is.na(dados_treino$Age))
dados_treino_sem_NA <- filter(dados_treino2, dados_treino$Age > 0)
```

Para prever os valores NA da variável Age, vamos criar um algoritmo de regressão linear para prever a idade dos passageiros com base nas outras variáveis (quantidade de filhos, esposa, classe, etc.)

Criamos um modelo de regressão linear
```{r}
modeloAvaliarIdade <- lm(Age ~ ., data = dados_treino_sem_NA)
modeloAvaliarIdade
```

Previsão de valores onde Age = NA, com base nas idades informadas das pessoas utilizando regressão linear:
```{r}
previsaoIdades <- predict(modeloAvaliarIdade, dados_treino_com_NA)
```

Criamos uma nova tabela retirando a variável age pois só continha valores NA
```{r}
dados_treino_NA_Previsto <- transmute(dados_treino_com_NA,
                                      Survived, Pclass, Sex, SibSp, Parch)
```

Incluir a variável onde estão as previsões das idades, alterando o nome para Age
```{r}
dados_treino_NA_Previsto <- mutate(dados_treino_NA_Previsto,
                                   Age = previsaoIdades)
```

Alterar a posição da variável Age no dataset, para que fique logo após a variável Sex. Desta forma, os datasets ficarão estruturados da mesma forma e será possível juntá-los sem nenhum problema:
```{r}
dados_treino_NA_Previsto <- dados_treino_NA_Previsto %>% relocate(Age, .after = Sex)
```

Juntar as tabelas em um dados_treino_tratado
```{r}
dados_treino2 <- bind_rows(dados_treino_NA_Previsto, dados_treino_sem_NA)
```

Contando novamente os valores NA e verificando que agora não temos mais valores NA:
```{r}
print("Quantidade de dados NA:")
sum(is.na(dados_treino2))
```

Vamos agora criar um novo modelo para fazer as previsões se os passageiros sobreviveram ou não. 

Criar um modelo de regressão linear para previsão de sobreviventes:
```{r}
modeloAnaliseSobreviventes <- lm(Survived ~ ., data = dados_treino2)
summary(modeloAnaliseSobreviventes)
```

Como podemos ver na avaliação do modelo acima: onde Pr(>|T|) possui valores muito baixos, significa que a relação com a predição do modelo é muito alta. 

A variável Age é uma das variáveis mais preditores e como já tivemos problemas com dados faltantes da variável Age do dataset de treino, podemos ter o mesmo problema com o dataset de teste. Então, vamos criar dois modelos preditivos:

1 modelo para prever dados onde o dataset analisado possui valores NA na variável Age
1 modelo para prever dados onde o dataset analisado não possui valores NA na variável Age

Criar um modelo de regressão linear para prever dados em um dataset que possua valores NA na variável Age
```{r}
modeloAnaliseSobreviventesSemAge <- lm(Survived ~ Pclass + Sex + SibSp, data = dados_treino2)
summary(modeloAnaliseSobreviventesSemAge)
```

Podemos ver as variaveis sex, age e pclass são mais significantes. 

Vamos então organizar os dados de teste, para fazer as predições:
```{r}
str(dados_teste)
dados_teste2 <- select(dados_teste, PassengerId, Pclass, Sex, Age, SibSp, Parch)
```

Separar os dados de teste em uma tabela com valores NA e outra sem valores Na na coluna Age
```{r}
dados_teste_com_NA <- filter(dados_teste2, is.na(dados_teste2$Age))
dados_teste_sem_NA <- filter(dados_teste2, dados_teste2$Age > 0)
```

Previsao se sobreviveu ou não para os dados sem valor NA em Age:
```{r}
previsaoSurvivers1 <- predict(modeloAnaliseSobreviventes, dados_teste_sem_NA)
```

Previsao se sobreviveu ou não para os dados com valor NA em Age
```{r}
previsaoSurvivers2 <- predict(modeloAnaliseSobreviventesSemAge, dados_teste_com_NA)
```

Vamos arredondar os valores para mostrar apenas 0 ou 1 (não sobreviveu ou sobreviveu, respectivamente):
```{r}
previsaoSurvivers1 <- round(previsaoSurvivers1, 0)
```

Vamos selecionar as colunas do dataset onde não tínhamos dados NA:
```{r}
dados_teste_sem_NA <- bind_cols(dados_teste_sem_NA, 
                                Survived = previsaoSurvivers1)
```

E selecionamentos apenas as variáveis PassengerId e Survived (Kaggle precisa dos dados dessa maneira para fazer os testes):
```{r}
dados_teste_sem_NA <- select(dados_teste_sem_NA, 
                             PassengerId, Survived)
```

Vamos arredondar os valores para mostrar apenas 0 ou 1 (não sobreviveu ou sobreviveu, respectivamente):
```{r}
previsaoSurvivers2 <- round(previsaoSurvivers2, 0)
```

Vamos selecionar as colunas do dataset onde tínhamos dados NA:
```{r}
dados_teste_com_NA <- bind_cols(dados_teste_com_NA, 
                                Survived = previsaoSurvivers2)
```

E selecionamentos apenas as variáveis PassengerId e Survived (Kaggle precisa dos dados dessa maneira para fazer os testes):
```{r}
dados_teste_com_NA <- select(dados_teste_com_NA, 
                             PassengerId, Survived)
```

Juntamos agora os dois datasets:
```{r}
PrevisaoSurvivers <- bind_rows(dados_teste_com_NA, dados_teste_sem_NA)
```

E organizamos em ordem crescente com base na variável PassengerId:
```{r}
PrevisaoSurvivers <- arrange(PrevisaoSurvivers, PassengerId)
```

Vamos gerar um arquivo .csv:
```{r}
write.csv(PrevisaoSurvivers, "previsao.csv", row.names = FALSE)
```

Após a predição, o dado foi inserido no Kaggle e consegui uma taxa de acertos de 0,7606, ou seja, 76% de precisão.

Fiquei muito feliz pois foi meu primeiro exercício realizado!

Vamos fazer uma previsão se eu sobreviveria ao Titanic? Vamos "chutar" que eu estaria na segunda classe:
```{r}
dadosTeste = data.frame(Pclass = c(2),
                        Sex = c("male"),
                        Age = c(30),
                        SibSp = c(1),
                        Parch = c(1),
                        Fare = c(NA),
                        Embarked = c(NA))
```

```{r}
testePredicao <- predict(modeloAnaliseSobreviventesSemAge, dadosTeste)
testePredicao
```

É.. Acho que eu não sobreviveria.. =´(
