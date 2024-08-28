# Data Side of the Moon 
## Decoding Pink Floyd’s Legacy

![Data Side of the Moon](media/DSOTM.png)

## Introduction

How can computer "precieve" music? How we, human, understand music? From a higher perspective, the process of precieve music is consist of 3 main steps: we hear(encode), we understand, and (if we want) we sing(decode) and then we hear (again). Base on this idea, we implement a neural network based on convolutional autoencoder architecture, and then explore the learning representation of songs.

The key idea of this project is using unsupervised learning to explore music, which should be distinguished from music genres classification tasks which we consider has its own limitation due to the labels(genres) are limited by human ourselves, as we believe that generes always falls behind music.

## Model "Echoes"

强调 reconstruction loss，如果不用decoder的话，老姐说的有道理；突出重点

*"Echoes" is my top favorite Pink Floyd song, and the name "Echoes" it self is a good metaphor for the autoencoder model which kind like reflections between songs.*

the bottleneck of the autoencoder is the "Data Side" of a given song, in this work we try our model on Pink Floyd's songs, and we call it "Data Side of the Moon".

## Discussion and future work

- open world (classify things to unknown classes)
