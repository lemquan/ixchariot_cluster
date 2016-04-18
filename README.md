#ixchriot_cluster
We perform logistic regression and infinite GMM on simulated data. The simulated data consists of 2 hosts and 5 receievers whereby a full mesh traffic to simulate the traffic of Bit Torrent downloads, Netflix, HTTP downloads, and Facebook. We allow the simulation to run for 5 hours. 

## Features 
Our dataset consists of traffic flow linked to a particular traffic mentioned above. The dataset uses all the fields gathered from IXChariot's NetFlow like data capture as well as some additional features self created. Those features are mostly ratios. We also took the log of the data set as it is in large bytes, and we perform a standard z-normalisation. Another important processing step we took was downnsampling. We captured the data at a high rate so that there is linear dependence amongst the rows. We randomly select a row for every 10 rows and also removed any rows that is entirely 0s. 

## Learning the Model 
We decided on two supervised approaches: clustering and logistic regression. We show that clustering did not perform well. Logistic regression achieves a 96% accuracy with cross-validation for parameter tuning. For CV, we use a time-series chaining where the validation set consists of future data. 
