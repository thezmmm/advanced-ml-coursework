## Support Vector Machines

### a

- The decision surface

  The line equidistant from the two nearest points of the same kind, i.e., the line satisfying ```wTx + b = 0```.

- The weight vector w 

  Perpendicular to the decision plane, pointing towards the positive class (+1) direction.

- The support vectors 

  The point closest to the decision plane 

- The margin 

  Distance between two support vector hyperplanes, ```2/||w||```

![2a](E:\Content\University of Helsinki\Advanced Course in Machine Learning\Exercise2\2a.png)

### b

1. Lagrangian

$$
\Lambda( w , b , \xi, \alpha, \beta)=\frac{1} {2} \| w \|^{2}+C \sum_{n} \xi_{n}-\sum_{n} \alpha_{n} [ y_{n} ( w^{T} x_{n}+b )-1+\xi_{n} ]-\sum_{n} \beta_{n} \xi_{n}
$$

2. Differentiate the original variable and set it to zero

$$
\nabla_{w} \Lambda=w-\sum_{n} \alpha_{n} y_{n} x_{n}=0 \implies\boxed{w=\sum_{n} \alpha_{n} y_{n} x_{n}}
$$

$$
\nabla_{b} \Lambda=-\sum_{n} \alpha_{n} y_{n}=0 \implies\boxed{\sum_{n} \alpha_{n} y_{n}=0}
$$

$$
\frac{\partial\Lambda} {\partial\xi_{n}}=C-\alpha_{n}-\beta_{n}=0 \implies\beta_{n}=C-\alpha_{n}
$$

3. Substituting into the Lagrangian problem yields the duality problem
   $$
   \cfrac{1} {2} \| w \|^{2}=\cfrac{1} {2} \sum_{i , j} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}
   $$

   $$
   \sum_{n} \alpha_{n} y_{n} ( w^{T} x_{n}+b )=\sum_{n} \alpha_{n} y_{n} w^{T} x_{n}+b \underbrace{\sum_{n} \alpha_{n} y_{n}}_{=0}=\sum_{i , j} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}
   $$
$$
-\sum_{n} \beta_{n} \xi_{n}=-\sum_{n} (C-\alpha_n) \xi_{n}
$$

$$
\Lambda=\frac{1} {2} \sum_{i , j} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}+\sum_{n} \alpha_{n}-\sum_{i , j} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}=\sum_{n} \alpha_{n}-\frac{1} {2} \sum_{i , j} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j}
$$

   

Therefore, the dual maximization problem is:
$$
\boxed{\operatorname*{max}_{\alpha} \left[ \sum_{i} \alpha_{i}-\frac{1} {2} \sum_{i , j} \alpha_{i} \alpha_{j} y_{i} y_{j} x_{i}^{T} x_{j} \right] , \quad\mathrm{s . t .} \; 0 \leq\alpha_{i} \leq C , \sum_{i} \alpha_{i} y_{i}=0}
$$
