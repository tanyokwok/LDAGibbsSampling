# LDAGibbsSampling
This is a project witch implements topic model lda and uses lda in real application.
![image](https://github.com/fortianyou/LDAGibbsSampling/blob/master/lda.png)
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"> </script>
- 令$K, V, D$分别为主题，词汇，文档个数
- 对文档中的每个词已观测词$w_i=t$，$p(z_i=k|\overrightarrow{z}_{\neg i},\overrightarrow{w}_{i})\propto p(z_i=k,w_i=t|\overrightarrow{z}_{\neg i},\overrightarrow{w}_{\neg i})$
  - 每个$w_i$的主题$z_i$应该按照上述的分布来抽样，$i$表示词汇在文档中的位置。
  - 但是在初始化过程中，我们为了简便可以随机为每个$w$，随机分配一个$z$。
$$p(topic|doc)=\hat{\theta}_{mk}=\frac{n_{m,\neg i}^{(k)}+\alpha_k}{\sum_{k=1}^{K}{(n_{m,\neg i}^{(k)}+\alpha_k)}}$$
$$p(word|topic)=\hat{\varphi}_{kt}=\frac{n_{k,\neg i}^{(t)}+\beta_t}{\sum_{t=1}^{V}{(n_{k,\neg i}^{(t)}+\beta_t)}}$$
其中$n_{m,\neg i}^{(k)}$在文档$m$中去除了$w_i$之后的属于主题$k$的词个数；$n_{k,\neg i}^{(t)}$表示去除了$w_i$之后输入词项$t$属于主题$k$的次数。
$$p(z_i=k|\overrightarrow{z}_{\neg i},\overrightarrow{w}_{i}) \propto \hat{\theta}_{mk}·\hat{\varphi}_{kt} $$

- 几个参数
 - BURN_IN: 在Gibbs迭代收敛之前的迭代次数。因此，每次重采样都要先迭代BURN_IN次使得马氏链达到平稳分布（我们相信当迭代次数足够多之后马氏链达到平稳分布，尽管不一定如此）
 - SAMPLE_LAG: 表示抽样的间隔，样本抽样应该在马氏链达到平稳之后才进行。也就是算法迭代次数超过BURN_IN之后。
 - $\alpha$: 生成文档主题分布 $\theta$ 的狄利克雷分布先验参数
 - $\beta$: 生成主题生成词分布 $\varphi$ 的狄利克雷分布先验参数

