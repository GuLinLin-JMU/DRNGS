# A genomic selection method based on artificial intelligence. <br>
![](https://halobi.com/wp-content/uploads/2016/08/r_logo.png "R logo")
![](https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcSvCvZWbl922EJkjahQ5gmTpcvsYr3ujQBpMdyX-YG99vGWfTAmfw "linux logo")
![](https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcS3RzhXKSfXpWhWhvClckwi1Llj1j3HvjKpjvU8CQv4cje23TwS "windows logo")
## Brief introduction <br>
Genomic selection (GS) is a new method of selective breeding using single 
nucleotide polymorphism (SNP) markers throughout the whole genome. At present, 
there have been many GS models, but the conventional GS models are linear 
statistical models, which cannot capture the complex relationships among the 
genotypes, which limits the prediction accuracy of the models. Therefore, we 
developed a new deep learning GS method, named DRNGS. The characteristics of the 
new method are as follows :(1) deep residual network is used to predict GEBV, 
which can capture the complex relationships within genotypes and improve the 
prediction accuracy; (2) The strategies of convolution and pooling are adopted to 
reduce the complexity of high-dimensional genotype data and accelerate the 
computation speed; (3) The batch normalization layer (BN) is introduced into the 
model, which speeds up the convergence rate of the model. We provided a new 
solution to the problems of slow convergence speed and long computation time in 
deep learning.
## Version and download <br>
* [Version 0.1.0](https://github.com/GuLinLin-JMU/DRNGS/blob/master/DRNGS_0.1.0.tar.gz) -First version released on Nov, 14th, 2021<br>
## Running build-in data
```R
library("DRNGS")
data(wheat_example)
Markers <- wheat_example$Markers
y <- wheat_example$y
cvSampleList <- cvSampleIndex(length(y),10,1)
cvIdx <- 1
trainIdx <- cvSampleList[[cvIdx]]$trainIdx
testIdx <- cvSampleList[[cvIdx]]$testIdx
trainMat <-Markers[trainIdx,]
trainPheno <- y[trainIdx]
validIdx <- sample(1:length(trainIdx),floor(length(trainIdx)*0.1))
validMat <- trainMat[validIdx,]
validPheno <- trainPheno[validIdx]
trainMat <- trainMat[-validIdx,]
trainPheno <- trainPheno[-validIdx]
testMat <- Markers[testIdx,]
testPheno <- y[testIdx]
Res_kernel <- c("1*18")
Res_stride <- c("1*1")
Res_num_filter <- c(8)
Respool_type <- c("max")
Respool_kernel <- c("1*4")
Respool_stride <- c("1*4")
Res_layer_num <- c(1)
Res_act_type <- c("relu")
block_kernel <- c("1*17")
block_stride <- c("1*1")
block_pad <- c("0*8")
block_num_filter <- c(8)
block_act_type <- c("relu")
fullayer_num_hidden <- c(32,1)
fullayer_act_type <- c("relu")
drop_float <- c(0.2,0.1,0.05)
ResFrame <- list(Res_kernel=Res_kernel, Res_stride=Res_stride, Res_num_filter=Res_num_filter,
                 Res_act_type=Res_act_type, block_kernel=block_kernel,block_stride=block_stride, 
                 block_pad=block_pad,block_num_filter=block_num_filter, block_act_type=block_act_type,
                 Respool_type=Respool_type, Respool_kernel=Respool_kernel,Respool_stride=Respool_stride, 
                 fullayer_num_hidden=fullayer_num_hidden, fullayer_act_type=fullayer_act_type,
                 drop_float=drop_float, Res_layer_num=Res_layer_num)
markerImage = paste0("1*",ncol(trainMat))
```
## Training DRNGS model
```R
train_model <- DRNGS(trainMat = trainMat,trainPheno = trainPheno, validMat = validMat,
                       validPheno = validPheno, markerImage = markerImage, 
                       ResFrame = ResFrame,device_type = "gpu", gpuNum = 0, 
                       eval_metric = "mae",num_round = 100,array_batch_size= 30, 
                       learning_rate = 0.01,momentum = 0.5,wd = 0.00001, randomseeds = 0, 
                       initializer_idx = 0.01,verbose = TRUE)
```
## Prediction 
```R
predscores <- predict.DRNGS(GSModel = train_model,testMat = testMat,
              markerImage = markerImage )
```
## How to access help
If users have any bugs or issues or any suggestions are available, feel free to contact:<br>
Linlin Gu: linlin-gu@outlook.com <br>
Prof. Ming Fang: fangming618@126.com<br>
