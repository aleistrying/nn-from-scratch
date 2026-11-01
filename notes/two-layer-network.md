# Two-layer network notes

## Architecture

- $n_x$: number of input features.
- $m$: number of examples, stored as columns.

| Layer           | Neurons | Activation |
| --------------- | ------: | ---------- |
| Input           |   $n_x$ | —          |
| Hidden, layer 1 |       5 | tanh       |
| Output, layer 2 |       1 | sigmoid    |

## Shapes

General shape:

- $W^{[l]}$ = ($n^{[l]}$, $n^{[l-1]}$)
- $b^{[l]}$ = ($n^{[l]}$, $1$)
- $X$ = $A^{[0]}$ = ($n_x$, $m$)

| Item      | Shape        |
| --------- | ------------ |
| $A^{[0]}$ | ($n_x$, $m$) |
| $W^{[1]}$ | ($5$, $n_x$) |
| $b^{[1]}$ | ($5$, $1$)   |
| $A^{[1]}$ | ($5$, $m$)   |
| $W^{[2]}$ | ($1$, $5$)   |
| $b^{[2]}$ | ($1$, $1$)   |
| $A^{[2]}$ | ($1$, $m$)   |

## Forward pass

$$
Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}
$$

$$
A^{[1]} = \tanh(Z^{[1]})
$$

$$
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}
$$

$$
A^{[2]} = \text{sigmoid}(Z^{[2]})
$$

## Loss

Here $a^{(i)}$ means the output-layer prediction for example $i$.

$$
J = -\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log a^{(i)}
+ (1-y^{(i)})\log(1-a^{(i)})\right]
$$

## Backward pass: batch formulas

These are the batched versions of the single-connection derivations below. Examples are columns. Average once when calculating each weight or bias gradient.

### Output layer

$$
dZ^{[2]} = A^{[2]} - Y
$$

$$
dW^{[2]} = \frac{1}{m}dZ^{[2]}(A^{[1]})^T
$$

$$
db^{[2]} = \frac{1}{m}\sum_{i=1}^{m}dZ^{[2](i)}
$$

### Hidden layer

$$
dA^{[1]} = (W^{[2]})^T dZ^{[2]}
$$

$$
dZ^{[1]} = dA^{[1]} \odot \left(1-(A^{[1]})^2\right)
$$

$$
dW^{[1]} = \frac{1}{m}dZ^{[1]}X^T
$$

$$
db^{[1]} = \frac{1}{m}\sum_{i=1}^{m}dZ^{[1](i)}
$$

- In each bias sum, $i$ identifies an example and $dZ^{[l](i)}$ is its column in $dZ^{[l]}$. Add the $m$ columns and divide by $m$, keeping one column: one value per neuron.
- $dZ$ stores each example's loss derivative; $dW$ and $db$ are derivatives of the mean loss $J$.
- $\odot$ means element-by-element multiplication, not matrix multiplication. The activation square is also element by element.
- Compute all gradients before updating either layer's weights.

| Quantity             | Shape     |
| -------------------- | --------- |
| $dZ^{[2]}$           | $(1,m)$   |
| $dW^{[2]}$           | $(1,5)$   |
| $db^{[2]}$           | $(1,1)$   |
| $dA^{[1]}, dZ^{[1]}$ | $(5,m)$   |
| $dW^{[1]}$           | $(5,n_x)$ |
| $db^{[1]}$           | $(5,1)$   |

## Checks and explanation

Here $L$ is one example's loss; $J$ is the mean across examples. The chains below describe one connection at a time; the batch section combines these derivatives across examples.

Trace the dependencies using the chain rule:

**Output-layer weights:** $L \rightarrow a^{[2]} \rightarrow z^{[2]} \rightarrow w^{[2]}$.

**Hidden-layer weights:** $L \rightarrow a^{[2]} \rightarrow z^{[2]} \rightarrow a^{[1]} \rightarrow z^{[1]} \rightarrow w^{[1]}$.

$$
\frac{dL}{dw^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{dw^{[2]}}
$$

$$
\frac{dL}{dw^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{dw^{[1]}}
$$

$$
\frac{dL}{db^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{db^{[2]}}
$$

$$
\frac{dL}{db^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{db^{[1]}}
$$

Now find each derivative in the chain. In the loss and sigmoid sections, $a$ refers to the output activation; in the tanh section, it refers to the hidden activation.

### Derivative of loss

$$
\frac{dL}{da} = \frac{d}{da}L(y,a)
$$

$$
= \frac{d}{da}(-[y\log(a) + (1 - y)\log(1 - a)])
$$

$$
= -y \frac{d}{da}\log(a) - (1 - y)\frac{d}{da}\log(1 - a)
$$

$$
= -y \frac{1}{a} - (1 - y)\frac{1}{1-a} \times (-1)
$$

$$
=  -\frac{y}{a} + \frac{1 - y}{1 - a}
$$

### Derivative of sigmoid

$$
\frac{da}{dz} =  \frac{d}{dz}(g(z)) = \frac{d}{dz}(sigmoid(z))
$$

$$
=\frac{d}{dz}(\frac{1}{1 + e^{-z}})
$$

$$
= \frac{d}{dz}(1 + e^{-z})^{-1}
$$

$$
=-(1 + e^{-z})^{-2} \times \frac{d}{dz}(1 + e^{-z})
$$

$$
= -(1 + e^{-z})^{-2} \times e^{-z} \times -1
$$

can be rewritten like:

$$
\frac{da}{dz} = e^{-z}a^{2}
$$

since sigmoid is:

$$
a = \frac{1}{1+e^{-z}}
$$

$$
a(1+e^{-z}) = 1
$$

$$
a+ae^{-z} = 1
$$

$$
ae^{-z} = 1 - a
$$

we rewrite as:

$$
\frac{da}{dz} = a(1-a)
$$

### Derivative of tanh

$$
\frac{da^{[1]}}{dz^{[1]}} = \frac{d}{dz^{[1]}} a
$$

$$
=\frac{d}{dz^{[1]}}(g(z^{[1]}))
$$

$$
= \frac{d}{dz^{[1]}}(\tanh(z^{[1]}))
$$

$$
=\frac{d}{dz^{[1]}}\left(\frac{e^z-e^{-z}}{e^z+e^{-z}}\right)
$$

$$
=\text{sech}^2(z)
$$

$$
=1-\tanh^2(z)
$$

$$
=1-a^2
$$

### Derivative of the linear expressions

$$
\frac{dz^{[1]}}{dw^{[1]}} = \frac{d}{dw^{[1]}}(w^{[1]}x+b^{[1]})
$$

$$
\frac{dz^{[1]}}{dw^{[1]}} = x
$$

$$
\frac{dz^{[2]}}{dw^{[2]}} = \frac{d}{dw^{[2]}}(w^{[2]}a^{[1]}+b^{[2]})
$$

$$
\frac{dz^{[2]}}{dw^{[2]}} = a^{[1]}
$$

$$
\frac{dz^{[2]}}{da^{[1]}} = \frac{d}{da^{[1]}}(w^{[2]}a^{[1]}+b^{[2]})
$$

$$
\frac{dz^{[2]}}{da^{[1]}} =w^{[2]}
$$

### Derivative of biases

$$
\frac{dz^{[1]}}{db^{[1]}} = \frac{d}{db^{[1]}}(w^{[1]}x+b^{[1]})
$$

$$
\frac{dz^{[1]}}{db^{[1]}} = 0 + 1 = 1
$$

$$
\frac{dz^{[2]}}{db^{[2]}} = \frac{d}{db^{[2]}}(w^{[2]}a^{[1]}+b^{[2]})
$$

$$
\frac{dz^{[2]}}{db^{[2]}} = 0 + 1 = 1
$$

### Resulting formulas - on eexample

For these scalar steps, $a^{[1]}$ is one chosen hidden neuron's activation, $w^{[2]}$ is its connection to the output, and $x$ is one chosen input feature. $a^{[2]}$ is the output prediction. The hidden weight and bias refer to that same chosen hidden neuron.

#### Output-layer weight: $w^{[2]}$

$$
\frac{dL}{dw^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{dw^{[2]}}
$$

$$
\frac{dL}{dw^{[2]}} = (-\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}})\,a^{[2]}(1-a^{[2]})\,a^{[1]}
$$

$$
\frac{dL}{dw^{[2]}} = a^{[1]}\left(-\frac{ya^{[2]}(1-a^{[2]})}{a^{[2]}} + \frac{a^{[2]}(1-a^{[2]})(1-y)}{1-a^{[2]}}\right)
$$

$$
= a^{[1]}(-y(1-a^{[2]}) + a^{[2]}(1-y))
$$

$$
= a^{[1]}(-y+ya^{[2]} + a^{[2]} - ya^{[2]})
$$

$$
= a^{[1]}(a^{[2]} - y)
$$

#### Hidden-layer weight: $w^{[1]}$

$$
\frac{dL}{dw^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{dw^{[1]}}
$$

$$
= (-\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}})\,a^{[2]}(1-a^{[2]})\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x
$$

$$
= (a^{[2]}-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x
$$

$$
= (a^{[2]}-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x
$$

#### Output-layer bias: $b^{[2]}$

$$
\frac{dL}{db^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{db^{[2]}}
$$

$$
\frac{dL}{db^{[2]}} = (-\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}})\,a^{[2]}(1-a^{[2]})\,1
$$

$$
\frac{dL}{db^{[2]}} = \left(-\frac{ya^{[2]}(1-a^{[2]})}{a^{[2]}} + \frac{a^{[2]}(1-a^{[2]})(1-y)}{1-a^{[2]}}\right)
$$

$$
= (-y(1-a^{[2]}) + a^{[2]}(1-y))
$$

$$
= (-y+ya^{[2]} + a^{[2]} - ya^{[2]})
$$

$$
= a^{[2]} - y
$$

#### Hidden-layer bias: $b^{[1]}$

$$
\frac{dL}{db^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{db^{[1]}}
$$

$$
= (-\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}})\,a^{[2]}(1-a^{[2]})\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,1
$$

$$
= (a^{[2]}-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,1
$$

$$
= (a^{[2]}-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)
$$

#### Copied single-neuron bias derivation (reference)

$$
\frac{dL}{db} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{db}
$$

$$
\frac{dL}{db} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\,a(1-a)
$$

$$
= -\frac{a(1-a)y}{a} + \frac{a(1-a)(1-y)}{1-a}
$$

$$
= -y(1-a) + a(1-y)
$$

$$
= ya-y+a-ya
$$

$$
= a-y
$$
