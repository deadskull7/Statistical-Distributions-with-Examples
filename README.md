# Statistical-Distributions-with-Examples

### Normal Distribution
### Poison Distribution
### Binomial Distribution

**Mean:** The numerical average.
```
dataset.mean()
```

**Median:** The 50th percentile (50% of response are higher and lower).
```
dataset.median()
```

**Mode:** Returns a table. Can return multiple value.
```
dataset.mode()
```

---
### Measures of Spread
How data varies from typical value.

**Quantile:** Figure out the values of the 0th, 25th, 50th, 75th, and top percentile using the quantile function.

```
five_num = [dataset["subset"].quantile(0),   
            dataset["subset"].quantile(0.25),
            dataset["subset"].quantile(0.50),
            dataset["subset"].quantile(0.75),
            dataset["subset"].quantile(1)]

five_num # Will output these ^ values
```

**Describe:** Also return quantiles, but also returns mean, standard of deviation, min and max values.

```
dataset["subset"].describe();
```

**IQR:**  Interquartile Range. Distance between the 3rd and the 1st quartile. Describes the data.

**RegressionPlot:** Plot data and a linear regression model fit.
```
import pandas as pd
x, y = pd.Series(x, name="x_var"), pd.Series(y, name="y_var")
ax = sns.regplot(x=x, y=y, marker="+")
```
![](https://seaborn.pydata.org/_images/seaborn-regplot-3.png)

**TimeSeriesPlot:** Plot one or more timeseries with flexible representation of uncertainty.
```
gammas = sns.load_dataset("gammas")
ax = sns.tsplot(time="timepoint", value="BOLD signal",unit="subject", condition="ROI",data=gammas)
```
![](https://seaborn.pydata.org/_images/seaborn-tsplot-2.png)

**HeatMap:**
```
sns.heatmap(flights, annot=True, fmt="d")
```
![](https://seaborn.pydata.org/_images/seaborn-heatmap-5.png)

**CountPlot:** Show the counts of observations in each categorical bin using bars.
```
sns.factorplot(x="class", hue="who", col="survived",data=titanic, kind="count",size=4, aspect=.7);
```
![](https://seaborn.pydata.org/_images/seaborn-countplot-6.png)

**KdePlot:** Fit and plot a univariate or bivariate kernel density estimate.
```
import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
ax = sns.kdeplot(x)
```
![](https://seaborn.pydata.org/_images/seaborn-kdeplot-1.png)

**BarPlot:** Show point estimates and confidence intervals as rectangular bars.
```
sns.barplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=False)
```
![](https://seaborn.pydata.org/_images/seaborn-barplot-10.png)



**Variance:** The average of the squared deviations (differences) from the mean.  A numerical value used to indicate how widely individuals in a group vary.
```
dataset["subset"].var()
```

**Standard Deviation:** Extent of deviation from mean.
```
dataset["subset"].std()
```

**Median Absolute Deviation (MAD):**  Alternative measure of spread based on median if the mean cannot be trusted (or is not a true representation).

```
abs_median_devs = abs(dataset["subset"] - dataset["subset"].median())
abs_median_devs.median() * 1.4826 #Multiplied by scale factor for normal distribution
```

---
### Skewness

**Skew:** Measure of skew or asymmetry of dataset. More extreme number measure is more skewed.
```
dataset["subset"].skew()
```

---

## <a name="stat-infer"></a>Statistical Inference
Sample to guess the whole. The process of analyzing sample data to gain insight into the population from which the data was collected and to investigate differences between data samples.

### Point Estimates
Estimate of the population based on sample data.

**Sample Mean:** The average of a sample. Same function for `.mean()`.
---

### Sampling Distribution and The Central Limit Theorem

**Histogram:** Show data/information in "bins".
```
pd.DataFrame(dataset).hist(bins=58,
                          range=(17,75),
                          figsize=(7,7))
print( stats.skew(dataset) )
```

**Central Limit Theorem:**  The distribution of many sample means, known as a sampling distribution, will be normally distributed. For example, get 200 samples of your sample data and make a point estimate for each. These point estimates will have a normal distribution.
```
np.random.seed(10)

point_estimates = []         # Make empty list to hold point estimates

for x in range(200):         # Generate 200 samples, each with 500 sampled values
    sample = np.random.choice(a= population_ages, size=500)
    point_estimates.append( sample.mean() ) # keep the sample mean in the list

pd.DataFrame(point_estimates).plot(kind="density",  # Plot sample mean density
                                   figsize=(7,7),
                                   xlim=(41,45))   
```

---

### Confidence Intervals
A range of values above and below a point estimate that captures the true population parameter at some predetermined confidence level. Calculate a confidence interval by taking a point estimate and then adding and subtracting a margin of error to create a range.
```
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)
```

OR

```
stats.t.interval(alpha=0.95,              # Confidence level
                 df=24,                   # Degrees of freedom
                 loc=sample_mean,         # Sample mean
                 scale=sigma)             # Standard deviation estimate
```

**Margin of Error:** The amount allowed for miscalculation.
```
margin_of_error = z_critical * (stdev/math.sqrt(sample_size))
```



**T-critical:** Use when you do not have the standard deviation but only a sample standard deviation.
```
t_critical = stats.t.ppf(q = 0.975, df=24) #df is degrees of freedom (sample size minus 1)
                                           # q = confidence interval of 95 #
```

**Degrees of Freedom:** Sample size minus 1.

**T-Distribution:** A distribution that closely resembles the normal distribution but that gets wider and wider as the sample size falls.


---

## Statistical Hypothesis Testing: The T-Test
What is the likelihood as a percentage that something interesting is going on (when comparing data). Checks the null hypothesis.

**Null Hypothesis:** Assumes nothing interesting is going on between whatever variables you are testing.

### One-Sample T-Test
A one-sample t-test checks whether a sample mean differs from the population mean.

```
stats.ttest_1samp(a=dataset_sample,               # Sample data
                  popmean=dataset_whole.mean())   # All data mean
```

### Two-Sample T-Test
Investigates whether the means of two independent data samples differ from one another.
```
stats.ttest_ind(a=dataset_sample,
                b=dataset_whole,
                equal_var=False)    # Assume samples have equal variance?
```


### Type I & II Errors

**Type I:** False-positive.  A situation where you reject the null hypothesis when it is actually true.

**Type II:** False-negative. A situation where you fail to reject the null hypothesis when it is actually false.
