# 📝 Relatório do Candidato


👤 Identificação: **Elilúcio Teixeira Félix Filho**
    email: **eliluciofilho@gmail.com**
    Discente do Curso: **Engenharia de Software**
    Instituição: **Universidade Federal do Cariri**

 
## 1️⃣ Resumo da Arquitetura do Modelo

O modelo construído da CNN teve as seguintes camadas:

### layers.Conv2D(20, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
- camada convolucional para filtro de características da imagem

### layers.MaxPooling2D(pool_size=(3, 3))
- camada usada para "simplificar" os dados da camada anterior, deixando apenas o maior valor de trechos da imagem

### layers.Conv2D(20, kernel_size=(3, 3), activation="relu")
- segunda camada convolucional

### layers.Flatten()
- camada de conversão do formato dos dados para leitura efetiva

### layers.Dense(32, activation="relu")
- camada densa de neurônios, onde será feita a análise principal da CNN

### layers.Dense(32, activation="relu")
- outra camada densa para menor modelo final sem perda significativa de acurácia comparado a apenas uma camada densa com o mesmo número de neurônios

### layers.Dropout(0.15)
- camada de dropout para evitar overfitting

### layers.Dense(10, activation='softmax')
- última camada densa para output da escolha feita por softmax


## 2️⃣ Bibliotecas Utilizadas

tensorflow (>=2.12)


## 3️⃣ Técnica de Otimização do Modelo

A técnica de otimização utilizada na conversão de .h5 para .tflite foi a **Quantização de Faixa Dinâmica**, que faz algumas mudanças com o intuito de reduzir o tamanho do arquivo e haver uma velocidade de operações maior, como a conversão dos pesos usados na CNN de float32 para int8.


## 4️⃣ Resultados Obtidos

A CNN resultante apresentou uma acurácia de aproximadamente **98.47%** e perda de **4.81** no modelo .h5, com tamanhos de:

- .h5 - 314KB
- .tflite - 29,8KB

Seu tempo de treinamento foi de aproximadamente um minuto, utilizando 5 épocas, batch size de 128 e validation split de 0.1.
Foram realizados em torno de quarenta testes para chegar em uma precisão elevada sem um tamanho de arquivo extenso. O valor de dropout também foi avaliado, pois, com números maiores, deteriorava o funcionamento da CNN, e sem sua presença havia um leve overfitting.


### 5️⃣ Comentários Adicionais (Opcional)

O projeto foi prazeroso de se realizar, além de muito benéfico para solodificar os aprendizados do curso. Foi interessante ver em primeira mão o peso das alterações feitas nas camadas, como a velocidade de treinamento com batch siz,e a força dos filtros de Max Pooling, a maior eficácia da separação da camada densa, sem comentar nos parâmetros de treinamento em si.
Tive certa dificuldade quanto a ter confiança de quando o modelo estaria eficiente o suficiente e se a arquitetura permitia alta precisão sem demasiado tamanho na parte inicial dos testes, mas após um tempo o impacto de cada camada, parâmetros de treinamento etc. têm no modeo. Fiquei um pouco surpreso quando percebi que a divisão da camada densa diminui notoriamente o tamanho final do modelo sem uma queda na precisão.