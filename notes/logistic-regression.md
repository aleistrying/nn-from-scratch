# Logistic regression: my working notes

## Shapes and notation

For one example, I chose column vectors:

$$X \in \mathbb{R}^{1\times6}, \qquad W \in \mathbb{R}^{6\times1}$$

$$m = examples$$

## Forward pass

6 inputs feed one neuron:

$$x_1, x_2, x_3, x_4, x_5, x_6 \longrightarrow n^{[1]}$$

My weighted-sum notation:

$$x_1w^{[1]}_1 + x_2w^{[1]}_2 + \cdots + x_iw^{[1]}_i + b = z$$

$$[x_1, x_2, \cdots, x_i]@[w^{[1]}_1, w^{[1]}_2, \cdots, w^{[1]}_i]^T + b = z$$

My vectorized:

$$XW^{[1]} + b = Z^{[1]}$$

or

$$A^{[0]}W^{[1]} + b = Z^{[1]}$$

Adding a function for non-linearity

$$g(Z^{[1]}) = A^{[1]}$$

<!-- $$(m,6) \times (6,m) = (m,m)$$ -->

The activation function we will be using:

$$g(x) = \frac{1}{1+e^{-x}}$$

## Loss for a single example

$$L(y, ŷ) = -[y{\log}(ŷ)+(1-y){\log}(1 - ŷ)]$$

For ease of writting loss will be $$L$$ and $$dL$$ implies the derivative of the loss function

## Gradients and parameter updates

The gradients we calculate using derivatives.

$$w^{[l]} := w^{[l]} - \alpha \times {\text dw}^{[l]}$$
$$b^{[l]} := b^{[l]} - \alpha \times {\text db}^{[l]}$$

the two values we have to find $dw$ and $db$:

$$\frac{dL}{dw} = x(a - y)$$
$$\frac{dL}{db} = a - y$$

## Checks and explanation

Since we can't directly get dw from $dL$ we have to use the chain rule in our backwards computation (l $\rightarrow$ a $\dots$ a $\rightarrow$ z $\dots$ z $\rightarrow$ w):

$$\frac{dL}{dw} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{dw}$$

$$\frac{dL}{db} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{db}$$

now let's find the derivatives of each of those to calculate

### Derivative of loss

$$\frac{dL}{da} = \frac{d}{da}(Loss(y,a))$$
$$= \frac{d}{da}(-[y\log(a) + (1 - y)\log(1 - a)])$$
$$= -y \frac{d}{da}\log(a) - (1 - y)\frac{d}{da}\log(1 - a)$$
$$= -y \frac{1}{a} - (1 - y)\frac{1}{(1 - a)} \times {-1}$$
$$=  -\frac{y}{a} + \frac{1 - y}{1 - a}$$

### Derivative of sigmoid

$$\frac{da}{dz} =  \frac{d}{dz}(g(z)) = \frac{d}{dz}(sigmoid(z))$$
$$=\frac{d}{dz}(\frac{1}{1 + e^{-z}})$$
$$= \frac{d}{dz}(1 + e^{-z})^{-1}$$
$$=-(1 + e^{-z})^{-2} \times \frac{d}{dz}(1 + e^{-z})$$
$$= -(1 + e^{-z})^{-2} \times e^{-z} \times -1$$

can be rewritten like:
$$\frac{da}{dz} = e^{-z}a^{2}$$

since sigmoid is:
$$a = \frac{1}{(1+e^{-z})}$$
$$a(1+e^{-z}) = 1$$
$$a+ae^{-z} = 1$$
$$ae^{-z} = 1 - a$$

we rewrite as:
$$\frac{da}{dz} = a(1-a)$$

<!-- somehow -->
<!-- $$\frac{dL}{da} = y - a??$$ -->

### derivative of linear regression formula

$$\frac{dz}{dw} = \frac{d}{dw}(wx+b)$$
$$\frac{dz}{dw} = x$$

### Derivative of bias

$$\frac{dz}{db} = \frac{d}{db}(wx+b)$$
$$\frac{dz}{db} = 0 + 1 = 1$$

### Resulting formulas

calculate for dw

$$\frac{dL}{dw} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{dw}$$
$$\frac{dL}{dw} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\times(a(1-a))\times(x)$$
$$\frac{dL}{dw} = x(-\frac{(y)(a(1-a))}{a} + \frac{(a(1-a))(1 - y)}{1 - a})$$
$$= x(-y(1-a) + (a1 - y))$$
$$= x(-y+ya + a - ya)$$
$$= x(a - y)$$

now for bias

$$\frac{dL}{db} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{db}$$

$$\frac{dL}{db} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\times(a(1-a))\times(1)$$
$$= (-\frac{(a(1-a))(y)}{a} + \frac{(a(1-a))(1 - y)}{1 - a})$$
$$= -(1-a)(y) + (a)(1 - y)$$
$$= ya-y+a-ya$$
$$= a-y$$
