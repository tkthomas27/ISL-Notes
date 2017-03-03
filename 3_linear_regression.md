---
title: 'Chapter 3: Linear Regression'
pdf_document: default
output:
  html_notebook:
    pandoc_args:
    - --number-sections
    - --number-offset=2
    toc: yes
  html_document:
    toc: yes
  pdf_document:
    toc: yes
---
\setcounter{section}{3}


# Linear Regression

**Questions**

- Is there a relationship bewtween Y and X?
- How strong is the relationship between Y and X?
- Which variable in X is related to Y?
- How accurate is estimated effect?
- How accurate are predictions?
- Is the relatinoship linear?
- Are their interactions between the X variables (synergy)?

## Simple Linear Regression

Basic equation for regressing Y on X is:

$$
Y \approx \beta_0 + \beta_1 X
$$

Once we have estimates of parameters $\beta_0$ and $\beta_1$, they can be used to estimate:

$$
\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x
$$

### Estimating the Coefficients

The goal is to find a set of parameters that brings the line as close as possible to the data. Most common way is to minimize the least squares criterion.

Regression boils down to a battle between $y$s --- the predicted and the actual $y$. In a single variable linear regression, we can plot the predicted value $\hat{y}$ against the single $x$ variable (this should form a line). If we put this line over a graph of the actual values of $y$ agains the single $x$, then we can measure difference between the actual values of $y$ and the predicted values of $y$ (i.e., $\hat{y}$). This difference is the *residual*: $e_i = y_i - \hat{y}_i$. 

We use this residual to calculate the residual sum of squares (RSS):

$$
\begin{split}
RSS &= \sum e_i^2\\
RSS &= \sum (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2
\end{split}
$$

**The Importance of RSS**: to estimate the $\beta$s we want to minimize the RSS. Remember: RSS is a measure of the distance between our prediction of $\hat{y}$ and what $y$ actually is. Therefore, it makes sense that we want to make this difference as small as possible (i.e., minimize the difference). This is done via calculus (see appendix) but in a single variable situation it can be show that the minimum for each $\beta$ is:

$$
\begin{split}
\hat{\beta}_1 &= \frac{\sum^n_{i=1} (x_i - \bar{x})(y_i - \bar{y})}{\sum^n_{i=1} (x_i - \bar{x})^2}\\
\hat{\beta}_1 &= \bar{y} - \hat{\beta}_1 \bar{x}
\end{split}
$$

The bars over $y$ and $x$ (read as y-bar and x-bar) represent the mean (i.e., $\bar{y} \equiv \frac{1}{n} \sum y_i$)


### Assessing the Accuracy of the Coefficient

### Assessing Accuracy of the Model

## Multiple Linear Regression


## Other Considerations in Regression Model


## Comparison to KNN

## Calculus Appendix