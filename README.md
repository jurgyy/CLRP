## Implementing Contrastive Layer-wise Relevance Propagation with Innvestigate

Contrastive Layer-wise Relevance Propagation[1] or CLRP is a modification of standard Layer-wise Relevance Propagation[2] (LRP) with the goal of making the output (more) class discriminative. This notebook will use the LRP library Innvestigate[3] to attempt to implement CLRP.

A general overview of CLRP is as follows:

 1. Given an output neuron ![ $`y_j`$](https://latex.codecogs.com/gif.latex?%24y_j%24) which represent concept ![$`O`$](https://latex.codecogs.com/gif.latex?%24O%24) we try to construct a *dual* virtual concept ![$`\overline O`$](https://latex.codecogs.com/gif.latex?%24%5Coverline%20O%24) which represents the opposite concept of ![$`O`$](https://latex.codecogs.com/gif.latex?%24O%24).  
 2. This concept ![$`\overline O`$](https://latex.codecogs.com/gif.latex?%24%5Coverline%20O%24) can be represented in two different ways:  
   A. CLRP1: The concept is represented by the selected classes with weights ![$`\overline W = \{W^1, W^2, ..., W^{L-1}, W^L_{-j}\}`$](https://latex.codecogs.com/gif.latex?%24%5Coverline%20W%20%3D%20%5C%7BW%5E1%2C%20W%5E2%2C%20...%2C%20W%5E%7BL-1%7D%2C%20W%5EL_%7B-j%7D%5C%7D%24). Here ![$`W_{-j}`$](https://latex.codecogs.com/gif.latex?%24W_%7B-j%7D%24) means the weights connected to the output layer excluding the ![$`j`$-th](https://latex.codecogs.com/gif.latex?%24j%24) neuron.    
   B. CLRP2: The concept is represented by the selected classes with weights![$`\overline W = \{W^1, W^2, ..., W^{L-1}, W^L_{-j}\}`$](https://latex.codecogs.com/gif.latex?%24%5Coverline%20W%20%3D%20%5C%7BW%5E1%2C%20W%5E2%2C%20...%2C%20W%5E%7BL-1%7D%2C%20W%5EL_%7B-j%7D%5C%7D%24). Which means all the weights are the same, except for the last layer where the weights to neuron $`j`$ are negated.  
 4. (?) The score ![$`S_{y_j}`$](https://latex.codecogs.com/gif.latex?%24S_%7By_j%7D%24) of target class is uniformly redistributted to other classes.  
 5. ![$`R_{\text{LRP}} = f_{\text{LRP}}(X, W, S_{y_j})`$](https://latex.codecogs.com/gif.latex?%24R_%7B%5Ctext%7BLRP%7D%7D%20%3D%20f_%7B%5Ctext%7BLRP%7D%7D%28X%2C%20W%2C%20S_%7By_j%7D%29%24)
 6. Given the same input example ![$`X`$](https://latex.codecogs.com/gif.latex?%24X%24) LRP generates an explanation ![$`R_{\text{dual}} = f_{\text{LRP}}(X, \overline W, S_{y_j})`$](https://latex.codecogs.com/gif.latex?%24R_%7B%5Ctext%7Bdual%7D%7D%20%3D%20f_%7B%5Ctext%7BLRP%7D%7D%28X%2C%20%5Coverline%20W%2C%20S_%7By_j%7D%29%24) for the dual concept.  
 7. Then CLRP is defined as follows: ![$`R_{\text{CLRP}} = \max(0, R - R_{\text{dual}})`$](https://latex.codecogs.com/gif.latex?%24R_%7B%5Ctext%7BCLRP%7D%7D%20%3D%20%5Cmax%280%2C%20R%20-%20R_%7B%5Ctext%7Bdual%7D%7D%29%24)
 
 _________
 
 Here are some results from the CLRP paper which shows a very clear class discrimitative property. These results are from using VGG16 pre-trained on imagenet and applying the ![$`z^\beta`$](https://latex.codecogs.com/gif.latex?%24z%5E%5Cbeta%24)-rule in the first convolution layer and for all the other convulutional layers the ![$`z^+`$](https://latex.codecogs.com/gif.latex?%24z%5E&plus;%24)-rule. For more details read the paper.
 
 ![results](https://user-images.githubusercontent.com/22032197/53959686-a6670000-40e4-11e9-8de2-ce13d038f5c1.png)
 
 _________
 
 [1] [Gu, Jindong, Yinchong Yang, and Volker Tresp. "Understanding Individual Decisions of CNNs via Contrastive Backpropagation." Asian Conference on Computer Vision. Springer, Cham, 2018](https://arxiv.org/pdf/1812.02100.pdf)

 [2] [Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)

 [3] [iNNvestigate neural networks!](https://github.com/albermax/innvestigate/) - github repository
