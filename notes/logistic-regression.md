# Logistic regression: my working notes

## Shapes and notation

Examples are columns. There are six input features and one output neuron.

$$m = \text{number of examples}, \qquad n_x = n^{[0]} = 6, \qquad n^{[1]} = 1$$

For one example:

$$x \in \mathbb{R}^{n_x\times1}, \qquad W^{[1]} \in \mathbb{R}^{1\times n_x}, \qquad b^{[1]} \in \mathbb{R}^{1\times1}$$

For a batch:

$$X = A^{[0]} \in \mathbb{R}^{n_x\times m}$$
$$Z^{[1]}, A^{[1]}, Y \in \mathbb{R}^{1\times m}$$

Layer weights use the convention $W^{[l]} \in \mathbb{R}^{n^{[l]}\times n^{[l-1]}}$: rows are output neurons and columns are input features. The same weights are shared across all $m$ examples.

## Forward pass

6 inputs feed one neuron:

$$x_1, x_2, x_3, x_4, x_5, x_6 \longrightarrow n^{[1]}$$

My weighted-sum notation:

$$x_1w^{[1]}_1 + x_2w^{[1]}_2 + \cdots + x_iw^{[1]}_i + b = z$$

$$[w^{[1]}_1, w^{[1]}_2, \cdots, w^{[1]}_i]\begin{bmatrix}x_1\\x_2\\\vdots\\x_i\end{bmatrix} + b = z$$

My vectorized form:

$$W^{[1]}X + b^{[1]} = Z^{[1]}$$

or

$$W^{[1]}A^{[0]} + b^{[1]} = Z^{[1]}$$

Adding a function for non-linearity

$$g(Z^{[1]}) = A^{[1]}$$

The activation function we will be using:

$$g(x) = \frac{1}{1+e^{-x}}$$

## Loss for a single example

$$L(y, ŷ) = -[y{\log}(ŷ)+(1-y){\log}(1 - ŷ)]$$

I write the loss as $L$. A derivative such as $dL/da$ specifies which variable the loss is differentiated with respect to.

## Gradients and parameter updates

The derivations below are for a single example and one scalar weight $w$. Batch gradients must match the shapes of $W^{[1]}$ and $b^{[1]}$, respectively.

$$w^{[l]} := w^{[l]} - \alpha \times {\text dw}^{[l]}$$
$$b^{[l]} := b^{[l]} - \alpha \times {\text db}^{[l]}$$

the two values we have to find $dw$ and $db$:

$$\frac{dL}{dw} = x(a - y)$$
$$\frac{dL}{db} = a - y$$

## Checks and explanation

Since we can't directly get dw from $dL$ we have to use the chain rule in our backwards computation ($L \rightarrow a \rightarrow z \rightarrow w$):

$$\frac{dL}{dw} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{dw}$$

$$\frac{dL}{db} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{db}$$

Now find each derivative in the chain.

### Derivative of loss

$$\frac{dL}{da} = \frac{d}{da}L(y,a)$$
$$= \frac{d}{da}(-[y\log(a) + (1 - y)\log(1 - a)])$$
$$= -y \frac{d}{da}\log(a) - (1 - y)\frac{d}{da}\log(1 - a)$$
$$= -y \frac{1}{a} - (1 - y)\frac{1}{1-a} \times (-1)$$
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
$$a = \frac{1}{1+e^{-z}}$$
$$a(1+e^{-z}) = 1$$
$$a+ae^{-z} = 1$$
$$ae^{-z} = 1 - a$$

we rewrite as:
$$\frac{da}{dz} = a(1-a)$$

### Derivative of the linear expression

$$\frac{dz}{dw} = \frac{d}{dw}(wx+b)$$
$$\frac{dz}{dw} = x$$

### Derivative of bias

$$\frac{dz}{db} = \frac{d}{db}(wx+b)$$
$$\frac{dz}{db} = 0 + 1 = 1$$

### Resulting formulas

For the weight:

$$\frac{dL}{dw} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{dw}$$
$$\frac{dL}{dw} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\,a(1-a)\,x$$
$$\frac{dL}{dw} = x\left(-\frac{ya(1-a)}{a} + \frac{a(1-a)(1-y)}{1-a}\right)$$
$$= x(-y(1-a) + a(1-y))$$
$$= x(-y+ya + a - ya)$$
$$= x(a - y)$$

For the bias:

$$\frac{dL}{db} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{db}$$

$$\frac{dL}{db} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\,a(1-a)$$
$$= -\frac{a(1-a)y}{a} + \frac{a(1-a)(1-y)}{1-a}$$
$$= -y(1-a) + a(1-y)$$
$$= ya-y+a-ya$$
$$= a-y$$
