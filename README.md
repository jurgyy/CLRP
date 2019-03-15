## Implementing Contrastive Layer-wise Relevance Propagation with Innvestigate

Contrastive Layer-wise Relevance Propagation[1] or CLRP is a modification of standard Layer-wise Relevance Propagation[2] (LRP) with the goal of making the output (more) class discriminative. This notebook will use the LRP library Innvestigate[3] to attempt to implement CLRP.

A general overview of CLRP is as follows:

 1. Given an output neuron $y_j$ which represent concept $O$ we try to construct a *dual* virtual concept $\overline O$ which represents the opposite concept of $O$.  
 2. This concept $\overline O$ can be represented in two different ways:  
   A. CLRP1: The concept is represented by the selected classes with weights $\overline W = \{W^1, W^2, ..., W^{L-1}, W^L_{-j}\}$. Here $W_{-j}$ means the weights connected to the output layer excluding the $j$-th neuron.    
   B. CLRP2: The concept is represented by the selected classes with weights $\overline W = \{W^1, W^2, ..., W^{L-1}, -1 * W^L_{j}\}$. Which means all the weights are the same, except for the last layer where the weights to neuron $j$ are negated.  
 4. (?) The score $S_{y_j}$ of target class is uniformly redistributted to other classes.  
 5. $R_{\text{LRP}} = f_{\text{LRP}}(X, W, S_{y_j})$
 6. Given the same input example $X$ LRP generates an explanation $R_{\text{dual}} = f_{\text{LRP}}(X, \overline W, S_{y_j})$ for the dual concept.  
 7. Then CLRP is defined as follows: $R_{\text{CLRP}} = \max(0, R - R_{\text{dual}})$
 
 _________
 
 Here are some results from the CLRP paper which shows a very clear class discrimitative property. These results are from using VGG16 pre-trained on imagenet and applying the $z^\beta$-rule in the first convolution layer and for all the other convulutional layers the $z^+$-rule. For more details read the paper.
 
 ![results](https://user-images.githubusercontent.com/22032197/53959686-a6670000-40e4-11e9-8de2-ce13d038f5c1.png)
 
 _________
 
 [1] @article{gu2018understanding,
  title={Understanding Individual Decisions of CNNs via Contrastive Backpropagation},
  author={Gu, Jindong and Yang, Yinchong and Tresp, Volker},
  journal={arXiv preprint arXiv:1812.02100},
  year={2018}
}

 [2] @article{bach-plos15,
    author = {Bach, Sebastian AND Binder, Alexander AND Montavon, Gr{\'e}goire AND Klauschen, Frederick AND M{\"u}ller, Klaus-Robert AND Samek, Wojciech},
    journal = {PLoS ONE},
    publisher = {Public Library of Science},
    title = {On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation},
    year = {2015},
    month = {07},
    volume = {10},
    url = {http://dx.doi.org/10.1371%2Fjournal.pone.0130140},
    pages = {e0130140},
    number = {7},
    doi = {10.1371/journal.pone.0130140}
}

 [3] https://github.com/albermax/innvestigate/
