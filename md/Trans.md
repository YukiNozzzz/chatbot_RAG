# Attention Is All You Need


<table border="1" ><tr>
<td colspan="1" rowspan="1">Ashish Vaswani*Google Brain</td>
<td colspan="1" rowspan="1">Noam Shazeer*Google Brain</td>
<td colspan="1" rowspan="1">Niki Parmar*Google Research</td>
<td colspan="1" rowspan="1">Jakob Uszkoreit*Google Research</td>
</tr><tr>
<td colspan="1" rowspan="1">avaswani@google.com</td>
<td colspan="1" rowspan="1">noam@google.com</td>
<td colspan="1" rowspan="1">nikip@google.com</td>
<td colspan="1" rowspan="1">usz@google.com</td>
</tr></table>


<table border="1" ><tr>
<td colspan="1" rowspan="1">Llion Jones*</td>
<td colspan="1" rowspan="1">Aidan N.Gomez*↑</td>
<td colspan="1" rowspan="1">Łukasz Kaiser*</td>
</tr><tr>
<td colspan="1" rowspan="1">Google Research</td>
<td colspan="1" rowspan="1">University of Toronto</td>
<td colspan="1" rowspan="1">Google Brain</td>
</tr><tr>
<td colspan="1" rowspan="1">1lion@google.com</td>
<td colspan="1" rowspan="1">aidan@cs.toronto.edu</td>
<td colspan="1" rowspan="1">lukaszkaiser@google.com</td>
</tr></table>

Illia Polosukhin*

illia.polosukhin@gmail.com

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture,the Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles,by over 2 BLEU.On the WMT 2014 English-to-French translation task,our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

## 1 Introduction

Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31,21,13].

*Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor.Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

Work performed while at Google Brain.

#Work performed while at Google Research.

31st Conference on Neural Information Processing Systems (NIPS 2017),Long Beach, CA,USA.

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht,as a function of the previous hidden state ht-1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and conditional computation [26],while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduc-tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [20],ByteNet [15]and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models,the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [11]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization,textual entailment and learning task-independent sentence representations [4, 22, 23, 19].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28].

To the best of our knowledge,however,the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8].

## 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5,2,29].Here,the encoder maps an input sequence of symbol representations$(x_{1},\cdots,x_{n})$to a sequence of continuous representations z=$(z_{1},\cdots,z_{n}).$. Given z, the decoderthe geerates an output sequence (y1,···,ym) of symbols oneelemenat a time. At each step the model is auto-regressive [9], consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise,fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,respectively.

### 3.1 Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack ofN=6identical layers.Each layer has two sub-layers.The first is a multi-head self-attenion mechanism, and the second is a simple, position-

<!-- Output Probabilities Softmax Linear Add &Norm Feed Forward Add & Norm Add & Norm Feed Multi-Head Forward Attention Nx Nx Add&Norm Add & Norm Multi-Head Masked Attention Multi-Head Attention Positional Encoding Positional Encoding Input Embedding Output Embedding Inputs Outputs (shifted right)  -->
![](https://textin-image-store-1303028177.cos.ap-shanghai.myqcloud.com/external/29e739ea7ac8150b)

#### Figure 1:The Transformer - model architecture.

wise fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers,followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x+ Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimensiondmodel=512.

Decoder: The decoder is also composed of a stack ofN=6identical layers. In addition to the two sub-layers in each encoder layer,the decoder inserts a third sub-layer,which performs multi-head attention over the output of the encoderstack. Similar to the encoder, we employ residual connections around each of the sub-layers,followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output,where the query,keys, values, and output are all vectors. The output is computed as a weighted sum of the values,where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk,and values of dimension du. We compute the dot products of the

<!-- MatMul SoftMax Mask (opt.) Scale MatMul Q K  -->
![](https://textin-image-store-1303028177.cos.ap-shanghai.myqcloud.com/external/781d621e369cbb32)

<!-- Multi-Head Attention Linear Concat Scaled Dot-Product Attention h Linear Linear Linear V K Q  -->
![](https://textin-image-store-1303028177.cos.ap-shanghai.myqcloud.com/external/543b731a1c63fcb2)

##### Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

query with all keys,divide each by$\sqrt{d_{k}},$,and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously,packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

Attentio$(Q,K,V)=sof\tanax(\frac{QK^{T}}{\sqrt{d_{k}}})V$ (1)

The two most commonly used attention functions are additive attention [2], and dot-product (multi-plictive) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of$\frac{1}{\sqrt{dk}}$.Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more spac-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk,the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by$\frac{1}{\sqrt{d_{k}}}.$

#### 3.2.2 Multi-Head Attention

Instead of performing a single attention function with dmodel-dimensional keys,values and queries,we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dy dimensions, respectively. On each of these projected versions of queries,keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values,as depicted in Figure 2.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q·$k=\sum_{i=1}^{dk}$ $q_{i}k_{i},$has mean 0 and variance dke.

MultiHea$d(Q,K,V)=Concat(head_{1},\cdots,headh)W^{o}$

$whereheadi=Attention(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})$

Where the projctios are parameter matrices$W_{i}^{Q}$E$:Red_{\max}\timesd_{k},W_{i}K\inRd_{madz}\timesd_{k},W_{i}^{V}\inRd_{modz}\timesd_{v}$and WO$\inRhd_{v}\timesd_{mode1}.$

In this work we employh=8parallel attention layers, or heads. For each of these we use $d_{k}=d_{v}=d_{modcl}/h=64.$Due to the reduced dimension of each head, the total computational cost issimiar to tatf single-head attention with full dimensionality.

#### 3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

·In "encoder-decoder attention" layers, the queries come from the previous decoder layer,and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [31,2,8].

·The encoder contains self-attention layers. In a self-attention layer all of the keys,values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previouslayer of the encoder.

·Similarly, self-attention layers i the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-egressive property. We implement this inside of scaled dotproduct attention by masking out (setting to -x) all values in the input of the softmax which correspond to illegal connections. See Figure 2.

### 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers,each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.This consists of two linear transformations with a ReLU activation in between.

$FFN(x)=\max(0,xW_{1}+b_{1})W_{2}+b_{2}$ (2)

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.The dimensionality of input and output is$d_{model}$=512,and the inner-layer has dimensionality $d_{ff}=2048.$

### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned lineartransfor-mation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [24]. In the embedding layers, we multiply those weights by vdmodel.

### 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence,we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the

#### Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.


<table border="1" ><tr>
<td colspan="1" rowspan="1">Layer Type</td>
<td colspan="1" rowspan="1">Complexity per Layer</td>
<td colspan="1" rowspan="1">SequentialOperations</td>
<td colspan="1" rowspan="1"> Maximum Path Length</td>
</tr><tr>
<td colspan="1" rowspan="1">Self-Attention</td>
<td colspan="1" rowspan="1">$O(n^{2}\cdotd)$</td>
<td colspan="1" rowspan="1">O(1)</td>
<td colspan="1" rowspan="1">O(1)</td>
</tr><tr>
<td colspan="1" rowspan="1">Recurrent</td>
<td colspan="1" rowspan="1">$O(n\cdotd^{2})$</td>
<td colspan="1" rowspan="1">O(n)</td>
<td colspan="1" rowspan="1">O(n)</td>
</tr><tr>
<td colspan="1" rowspan="1">Convolutional</td>
<td colspan="1" rowspan="1">$O(k\cdotn\cdotd^{2})$</td>
<td colspan="1" rowspan="1">O(1)</td>
<td colspan="1" rowspan="1">O(logk(n))</td>
</tr><tr>
<td colspan="1" rowspan="1">Self-Attention (restricted)</td>
<td colspan="1" rowspan="1">$O(r\cdotn\cdotd)$</td>
<td colspan="1" rowspan="1">O(1)</td>
<td colspan="1" rowspan="1">O(n/r)</td>
</tr></table>

bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings,learned and fixed [8].

In this work, we use sine and cosine functions of different frequencies:

$PE(p\cos,2i)=\sin(p\cos/10000^{2i}/d_{m\infty}dz)$

$PE_{(p\cos,2i+1)}=\cos(p\cos/10000^{2i/d_{mode})$

where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000·2π.We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k,$PE_{pos+k}$can be represented as a linear function of PEpos.

We also experimented with using learned positional embeddings [8] instead,and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

## 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolu-tional layers commonly used for mapping one variable-length sequence of symbol representations (x1,...,xn) to another sequence of equal length (z1,···,zn), with xi,zi E RRd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network.Learning long-range dependencies is a key challenge in many sequence transduction tasks.One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity,self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [31] and byte-pair [25] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in

the input sequence centered around the respective output posiion. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work.

A single convolutional layer with kernel widthk &lt;n does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers i the case of contiguous kernels,or O(logk(n)) in the case of dilated convolutions [15], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably,to$O(k\cdotn\cdotd+n\cdotd^{2}).$Even withk=n,however, the complexity f a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

## 5 Training

This section describes the training regime for our models.

### 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of abot 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000target tokens.

5.2 Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

### 5.3 Optimizer

We used the Adam optimizer [17] with$\beta_{1}=0.9,$ $\beta_{2}=0.98and$ $\epsilon=10^{-9}$We varied the learning rate over the course of training, according to the formula:

$lrate=d_{imoote}^{-0.5}\cdot\min(step_{-}num^{-0.5},step_{-}num\cdotwarmap_{-}steps^{-1.5})$ (3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmu$p_steps=4000.$

### 5.4 Regularization

We employ three types of regularization during training:

Residual Dropout We apply dropout [27] to the output of each sub-layer,before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model,we use a rate of $P_{drop}=0.1.$

#### Table 2:The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.


<table border="1" ><tr>
<td colspan="1" rowspan="2">Model</td>
<td colspan="2" rowspan="1">BLEU</td>
<td colspan="2" rowspan="1">Training Cost (FLOPs)</td>
</tr><tr>
<td colspan="1" rowspan="1">EN-DE</td>
<td colspan="1" rowspan="1">EN-FR</td>
<td colspan="1" rowspan="1">EN-DE</td>
<td colspan="1" rowspan="1">EN-FR</td>
</tr><tr>
<td colspan="1" rowspan="1">ByteNet [15]</td>
<td colspan="1" rowspan="1">23.75</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">$1.0\cdot10^{20}$</td>
</tr><tr>
<td colspan="1" rowspan="1">Deep-Att+PosUnk [32]</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">39.2</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">$1.4\cdot10^{20}$</td>
</tr><tr>
<td colspan="1" rowspan="1">GNMT+RL[31]</td>
<td colspan="1" rowspan="1">24.6</td>
<td colspan="1" rowspan="1">39.92</td>
<td colspan="1" rowspan="1">$2.3\cdot10^{19}$</td>
<td colspan="1" rowspan="1">$1.5\cdot10^{20}$</td>
</tr><tr>
<td colspan="1" rowspan="1">ConvS2S [8]</td>
<td colspan="1" rowspan="1">25.16</td>
<td colspan="1" rowspan="1">40.46</td>
<td colspan="1" rowspan="1">9.6·1018</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1">MoE [26]</td>
<td colspan="1" rowspan="1">26.03</td>
<td colspan="1" rowspan="1">40.56</td>
<td colspan="1" rowspan="1">2.0·1019</td>
<td colspan="1" rowspan="1">$1.2\cdot10^{20}$</td>
</tr><tr>
<td colspan="1" rowspan="1">Deep-Att+ PosUnk Ensemble [32]</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">40.4</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">8.0·1020</td>
</tr><tr>
<td colspan="1" rowspan="1">GNMT+ RL Ensemble [31]</td>
<td colspan="1" rowspan="1">26.30</td>
<td colspan="1" rowspan="1">41.16</td>
<td colspan="1" rowspan="1">1.8·1020</td>
<td colspan="1" rowspan="1">1.1·1021</td>
</tr><tr>
<td colspan="1" rowspan="1">ConvS2S Ensemble [8]</td>
<td colspan="1" rowspan="1">26.36</td>
<td colspan="1" rowspan="1">41.29</td>
<td colspan="1" rowspan="1">7.7·1019</td>
<td colspan="1" rowspan="1">1.2.$10^{21}$</td>
</tr><tr>
<td colspan="1" rowspan="1">Transformer(base model)</td>
<td colspan="1" rowspan="1">27.3</td>
<td colspan="1" rowspan="1">38.1</td>
<td colspan="2" rowspan="1">3.3·10$18$</td>
</tr><tr>
<td colspan="1" rowspan="1">Transformer (big)</td>
<td colspan="1" rowspan="1">28.4</td>
<td colspan="1" rowspan="1">41.0</td>
<td colspan="1" rowspan="1">$2.3\cdot10^{19}$</td>
<td colspan="1" rowspan="1"></td>
</tr></table>

Label Smoothing During training, we employed label smoothing of value$\epsilon_{ls}=0.1$[30]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

## 6 Results

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model(Transformer (big)in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0BLEU,establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate$P_{drop}=0.$1, instead of 0.3.

For the base models, we used a single model obtained by averaging the last 5 checkpoints,which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penaltyα=0.6[31].These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [31].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU 5.

### 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

In Table 3 rows (A),we vary the number of attention heads and the attention key and value dimensions,keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

5We used values of 2.8, 3.7,6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

#### Table 3: Variations on the Transformer architecture.Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.


<table border="1" ><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">N</td>
<td colspan="1" rowspan="1">dmodel</td>
<td colspan="1" rowspan="1">$d_{ff}$</td>
<td colspan="1" rowspan="1">h</td>
<td colspan="1" rowspan="1">$dk$</td>
<td colspan="1" rowspan="1">$d_{v}$</td>
<td colspan="1" rowspan="1">Pdrop</td>
<td colspan="1" rowspan="1">$\inls$</td>
<td colspan="1" rowspan="1">trainsteps</td>
<td colspan="1" rowspan="1">PPL(dev)</td>
<td colspan="1" rowspan="1">BLEU(dev)</td>
<td colspan="1" rowspan="1">params$\times10^{6}$</td>
</tr><tr>
<td colspan="1" rowspan="1">base</td>
<td colspan="1" rowspan="1">6</td>
<td colspan="1" rowspan="1">512</td>
<td colspan="1" rowspan="1">2048</td>
<td colspan="1" rowspan="1">8</td>
<td colspan="1" rowspan="1">64</td>
<td colspan="1" rowspan="1">64</td>
<td colspan="1" rowspan="1">0.1</td>
<td colspan="1" rowspan="1">0.1</td>
<td colspan="1" rowspan="1">100K</td>
<td colspan="1" rowspan="1">4.92</td>
<td colspan="1" rowspan="1">25.8</td>
<td colspan="1" rowspan="1">65</td>
</tr><tr>
<td colspan="1" rowspan="4">(A)</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">1</td>
<td colspan="1" rowspan="1">512</td>
<td colspan="1" rowspan="1">512</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.29</td>
<td colspan="1" rowspan="1">24.9</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4</td>
<td colspan="1" rowspan="1">128</td>
<td colspan="1" rowspan="1">128</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.00</td>
<td colspan="1" rowspan="1">25.5</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">16</td>
<td colspan="1" rowspan="1">32</td>
<td colspan="1" rowspan="1">32</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4.91</td>
<td colspan="1" rowspan="1">25.8</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">32</td>
<td colspan="1" rowspan="1">16</td>
<td colspan="1" rowspan="1">16</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.01</td>
<td colspan="1" rowspan="1">25.4</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="2">(B)</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">16</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.16</td>
<td colspan="1" rowspan="1">25.1</td>
<td colspan="1" rowspan="1">58</td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">32</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.01</td>
<td colspan="1" rowspan="1">25.4</td>
<td colspan="1" rowspan="1">60</td>
</tr><tr>
<td colspan="1" rowspan="7">(C)</td>
<td colspan="1" rowspan="1">2</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">6.11</td>
<td colspan="1" rowspan="1">23.7</td>
<td colspan="1" rowspan="1">36</td>
</tr><tr>
<td colspan="1" rowspan="1">4</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.19</td>
<td colspan="1" rowspan="1">25.3</td>
<td colspan="1" rowspan="1">50</td>
</tr><tr>
<td colspan="1" rowspan="1">8</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4.88</td>
<td colspan="1" rowspan="1">25.5</td>
<td colspan="1" rowspan="1">80</td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">256</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">32</td>
<td colspan="1" rowspan="1">32</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.75</td>
<td colspan="1" rowspan="1">24.5</td>
<td colspan="1" rowspan="1">28</td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">1024</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">128</td>
<td colspan="1" rowspan="1">128</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4.66</td>
<td colspan="1" rowspan="1">26.0</td>
<td colspan="1" rowspan="1">168</td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="2" rowspan="1">1024</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.12</td>
<td colspan="1" rowspan="1">25.4</td>
<td colspan="1" rowspan="1">53</td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4096</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4.75</td>
<td colspan="1" rowspan="1">26.2</td>
<td colspan="1" rowspan="1">90</td>
</tr><tr>
<td colspan="1" rowspan="4">(D)</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">0.0</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">5.77</td>
<td colspan="1" rowspan="1">24.6</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">0.2</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">4.95</td>
<td colspan="1" rowspan="1">25.5</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="2" rowspan="1">0.0</td>
<td colspan="1" rowspan="1">4.67</td>
<td colspan="1" rowspan="1">25.3</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1"></td>
<td colspan="2" rowspan="1">0.2</td>
<td colspan="1" rowspan="1">5.47</td>
<td colspan="1" rowspan="1">25.7</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1">(E)</td>
<td colspan="9" rowspan="1">positional embedding instead of sinusoids</td>
<td colspan="1" rowspan="1">4.92</td>
<td colspan="1" rowspan="1">25.7</td>
<td colspan="1" rowspan="1"></td>
</tr><tr>
<td colspan="1" rowspan="1">big</td>
<td colspan="1" rowspan="1">6</td>
<td colspan="1" rowspan="1">1024</td>
<td colspan="1" rowspan="1">4096</td>
<td colspan="2" rowspan="1">16</td>
<td colspan="1" rowspan="1"></td>
<td colspan="1" rowspan="1">0.3</td>
<td colspan="2" rowspan="1">300K</td>
<td colspan="1" rowspan="1">4.33</td>
<td colspan="1" rowspan="1">26.4</td>
<td colspan="1" rowspan="1">213</td>
</tr></table>

In Table 3 rows (B),we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D)that,as expected,bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E)we replace our sinusoidal positional encoding with learned positional embeddings [8], and observe nearly identical results to the base model.

## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention,replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks.We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments,corrections and inspiration.

### References

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473,2014.

[3] Denny Britz,Anna Goldie, Minh-Thang Luong,and Quoc V.Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong,and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk,and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078,2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357,2016.

[7] Junyoung Chung,Caglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolu-tional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2,2017.

[9] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850,2013.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im-age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.

[11] Sepp Hochreiter, Yoshua Bengio,Paolo Frasconi,and Jürgen Schmidhuber.Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[12]Sepp Hochreiter and Jürgen Schmidhuber.Long short-term memory. Neural computation,9(8):1735-1780,1997.

[13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[14] Łukasz Kaiser and Ilya Sutskever.Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.

[15] Nal Kalchbrenner, Lasse Espeholt,Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko-ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2,2017.

[16]Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush.Structured attention networks.In International Conference on Learning Representations, 2017.

[17] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR,2015.

[18] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722,2017.

[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding.arXiv preprint arXiv:1703.03130,2017.

[20] Samy Bengio Łukasz Kaiser. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.

[21] Minh-Thang Luong,Hieu Pham,and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025,2015.

[22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.

[23] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[24] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859,2016.

[25] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[26] Noam Shazeer, Azalia Mirhoseini,Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.arXiv preprint arXiv:1701.06538, 2017.

[27] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi-nov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958,2014.

[28] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N.D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors,Advances in Neural Information Processing Systems 28, pages 2440-2448.Curran Associates,Inc.,2015.

[29] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104-3112,2014.

[30] Christian Szegedy,Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567,2015.

[31] Yonghui Wu,Mike Schuster, Zhifeng Chen,Quoc V Le, Mohammad Norouzi, Wolfgang Macherey,Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144,2016.

[32] Jie Zhou,Yng Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199,2016.

