# Vision based inference of non-linear dynamical cloud timelapse
The target is to model timelapse videos of different weather scenarios. The weather is a fluid problem describable with the Navier-Stokes equations and thus its behavior is chaotic and non-linear.

## Abstract
Cold, hot, then windy, and suddenly a thunderstorm approaches. We all have experienced the chaotic and
very non-linear behavior of the weather and the challenge of predicting it. There are many well-established
numerical methods to predict the weather for several days into the future. But the local resolution is still not
satisfying enough.

In this bachelor thesis, we combine methods of computer vision and an analytic model discovery algo-
rithm to predict the cloud movement based on stationary timelapse using an autoencoder with the SINDy
algorithm. But first, we start with explaining the dynamics based on the Navier-Stokes equations and two an-
alytical methods to infer the dynamics: Dynamic Mode Decomposition (DMD) and Sparse Identification of
Non-linear Dynamics (SINDy). We showed that this classical SINDy method could model the system rela-
tively well, but it is computationally expensive and we do not have fundamental basis coordinates. Therefore
we proceed with our SINDy autoencoder deep neural network to get a low dimensional basis for SINDy. We
trained the autoencoder with our cloud timelapse videos to predict the future on the low dimensional space,
then decoded them back to the following frames. We showed that with the SINDy autoencoder model, we
could make reasonably good predictions on the training data set.

With this work, we introduced a new approach to predict cloud movements and potentially other weather
parameters. It could be used at airports or events to improve safety against sudden weather phenomena. In
future work, we imagine extracting three-dimensional data and thus getting a better spatial understanding of
the dynamics.

## Repository description
`Chapter-DMD_book/`: We first followed two books to learn the theory and to play with the sample code. These books are _Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems_ and _Data Driven Science and Engineering_.

`My_Models/Analytic/`: Analytical approaches using DMD and SINDy. This part is in matlab code. 

`My_Models/NeuralNetwork/`: Deep neural network approach using a SINDy autoencoder model to get a low dimensional representation of the video to predict the following state. 

`My_Models/Videos/`: We provide a link to low quality videos which we use for our autoencoder. 

## References
- _Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems(Authors: J. Nathan Kutz , Steven L. Brunton , Bingni W. Brunton and Joshua L. Proctor, https://doi.org/10.1137/1.9781611974508)_
- _Data Driven Science and Engineering (Authors: S.L. Brunton and J.N. Kutz, Cambridge University Press, http://www.databookuw.com/)_
- _Brian M. de Silva, Kathleen Champion, Markus Quade, Jean-Christophe Loiseau, J. Nathan Kutz, and Steven L. Brunton., (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. Journal of Open Source Software, 5(49), 2104, https://doi.org/10.21105/joss.02104_
- _B. Lusch Kathleen Championa and J. N. Kutz. Data-driven discovery of coordinates and governing
equations. Applied mathematics, 2019, arXiv:1904.02107v2)_
