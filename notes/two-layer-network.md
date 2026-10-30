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

$$ Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]} $$

$$ A^{[1]} = \tanh(Z^{[1]}) $$

$$ Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} $$

$$ A^{[2]} = \text{sigmoid}(Z^{[2]}) $$

## Loss

$$
J = -\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log a^{(i)}
+ (1-y^{(i)})\log(1-a^{(i)})\right]
$$

## Backward pass - batch formulas (draft)

These expressions are not verified yet. Check matrix dimensions and averaging separately from the single-example derivations below.

$$ dz^{[2]} = A^{[2]} - Y $$

$$ dw^{[2]} = dz^{[2]} A^{[1]} $$
TODO: derive the batch bias gradient for layer 2.

## Checks and explanation

Here $L$ is one example's loss; $J$ above is the mean across examples. The chains below describe dependencies; the matrix requires tracking individual gradients, this is for individual ones.

Trace the dependencies using the chain rule:
**Output-layer weights:** $L \rightarrow a^{[2]} \rightarrow z^{[2]} \rightarrow w^{[2]}$.

**Hidden-layer weights:** $L \rightarrow a^{[2]} \rightarrow z^{[2]} \rightarrow a^{[1]} \rightarrow z^{[1]} \rightarrow w^{[1]}$.

$$\frac{dL}{dw^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{dw^{[2]}}$$

$$\frac{dL}{dw^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{dw^{[1]}}$$

$$\frac{dL}{db^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{db^{[2]}}$$

$$\frac{dL}{db^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{db^{[1]}}$$

Now find each derivative in the chain. In the loss and sigmoid sections, $a$ refers to the output activation; in the tanh section, it refers to the hidden activation.

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

### Derivative of tanh

$$\frac{da^{[1]}}{dz^{[1]}} = \frac{d}{dz^{[1]}} a $$

$$=\frac{d}{dz^{[1]}}(g(z^{[1]}))$$

$$= \frac{d}{dz^{[1]}}(\tanh(z^{[1]}))$$

$$=\frac{d}{dz^{[1]}}\left(\frac{e^z-e^{-z}}{e^z+e^{-z}}\right)$$

$$=\text{sech}^2(z)$$

$$=1-\tanh^2(z)$$

$$=1-a^2$$

### Derivative of the linear expressions

$$\frac{dz^{[1]}}{dw^{[1]}} = \frac{d}{dw^{[1]}}(w^{[1]}x+b^{[1]})$$

$$\frac{dz^{[1]}}{dw^{[1]}} = x$$

$$\frac{dz^{[2]}}{dw^{[2]}} = \frac{d}{dw^{[2]}}(w^{[2]}a^{[1]}+b^{[2]})$$

$$\frac{dz^{[2]}}{dw^{[2]}} = a^{[1]}$$

<!--
$$\frac{dz^{[2]}}{da^{[1]}} = \frac{d}{da^{[1]}}(g(z^{[1]}))$$

$$= g'(z^{[1]})$$ -->

$$\frac{dz^{[2]}}{da^{[1]}} = \frac{d}{da^{[1]}}(w^{[2]}a^{[1]}+b^{[2]})$$
$$ \frac{dz^{[2]}}{da^{[1]}} =w^{[2]}$$

### Derivative of biases

$$\frac{dz^{[1]}}{db^{[1]}} = \frac{d}{db^{[1]}}(w^{[1]}x+b^{[1]})$$

$$\frac{dz^{[1]}}{db^{[1]}} = 0 + 1 = 1$$

$$\frac{dz^{[2]}}{db^{[2]}} = \frac{d}{db^{[2]}}(w^{[2]}x+b^{[2]})$$

$$\frac{dz^{[2]}}{db^{[2]}} = 0 + 1 = 1$$

### Resulting formulas - draft

The expressions below include copied logistic-regression steps and incomplete substitutions. They are retained for review, not presented as final two-layer gradients.

For $w^{[2]}$:

$$\frac{dL}{dw^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{dw^{[2]}}$$

$$\frac{dL}{dw} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\,a(1-a)\,x$$

$$\frac{dL}{dw} = x\left(-\frac{ya(1-a)}{a} + \frac{a(1-a)(1-y)}{1-a}\right)$$

$$= x(-y(1-a) + a(1-y))$$

$$= x(-y+ya + a - ya)$$

$$= x(a - y)$$

For $w^{[1]}$:

$$\frac{dL}{dw^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{dw^{[1]}}$$

$$= (-\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}})\,a^{[2]}(1-a^{[2]})\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x$$
$$= (a-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x$$
$$= (a^{[2]}-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x$$

For $b^{[2]}$:

$$\frac{dL}{db^{[2]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{db^{[2]}}$$

$$\frac{dL}{dw} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\,a(1-a)\,1$$

$$\frac{dL}{dw} = \left(-\frac{ya(1-a)}{a} + \frac{a(1-a)(1-y)}{1-a}\right)$$

$$= (-y(1-a) + a(1-y))$$

$$= (-y+ya + a - ya)$$

$$= a^{[2]} - y$$

For $b^{[1]}$:

$$\frac{dL}{db^{[1]}} = \frac{dL}{da^{[2]}}\times\frac{da^{[2]}}{dz^{[2]}}\times\frac{dz^{[2]}}{da^{[1]}}\times \frac{da^{[1]}}{dz^{[1]}}\times\frac{dz^{[1]}}{db^{[1]}}$$

$$= (-\frac{y}{a^{[2]}} + \frac{1 - y}{1 - a^{[2]}})\,a^{[2]}(1-a^{[2]})\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,1$$
$$= (a-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)\,x$$
$$= (a^{[2]}-y)\,w^{[2]}\,(1-\left(a^{[1]}\right)^2)$$

#### Copied single-neuron bias derivation (reference)

$$\frac{dL}{db} = \frac{dL}{da}\times\frac{da}{dz}\times\frac{dz}{db}$$

$$\frac{dL}{db} = (-\frac{y}{a} + \frac{1 - y}{1 - a})\,a(1-a)$$

$$= -\frac{a(1-a)y}{a} + \frac{a(1-a)(1-y)}{1-a}$$

$$= -y(1-a) + a(1-y)$$

$$= ya-y+a-ya$$

$$= a-y$$
