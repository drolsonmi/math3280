Following is weather data measured by our weather station here at Snow College. We are going to use this data to perform a linear regression. The goal will be to predict the relative humidity based on the other data.

* [Snow College Weather Data](https://raw.githubusercontent.com/drolsonmi/math3280/refs/heads/master/Notes/Data/Snow%20Weather_Daily.csv)

Documentation for weather data:
* `AirTF_Avg` = Average air temperature in Fahrenheit
* `AirTF_Max` = Maximum air temperature in Fahrenheit
* `AirTF_Min` = Maximum air temperature in Fahrenheit
* `RH_Avg` = Average daily relative humidity
* `WindGust` = Maximum wind gust (maximum singular wind speed)
* `AveWindSp` = Average wind speed
* `WindDir` = Wind direction
* `BP_inHg_Avg` = Barometric Pressure in inches of mercury (inHg)
* `Rain_Tot` = Total daily rainfall in inches
* `TdC_Min` = Minimum dew point temperature in Celsius
* `TdC_Max` = Maximum dew point temperature in Celsius

All other columns will not be needed for these assignments. So, load the data and drop all other columns. Also note that the first three rows are header rows. Keep the first as column headers, but remove the next two rows.

1. Is this dataset overdetermined or underdetermined?
2. Based on your answer from question 1, which of the following would you expect?

At this point, you will need to clean the data as we learned in MATH 3080.

3. Perform a linear regression just between Temperature (`AirTF_Avg`) and Relative Humidity (`RH_Avg`). Use the SVDs method to complete the linear regression. What is the value of the bias?
4. What is the value of the parameter?
5. Use the bias and parameter to predict the relative humidity if the temperature is $T$.
6. The correlation coefficient is found using the equation below. Find this correlation between temperature and humidity.
    $$r = \frac{1}{n-1}\sum \frac{x-\bar{x}}{s_x}\frac{y-\bar{y}}{s_y}$$


Notice that the regression is not perfect. That is because other variables come into play. We will now create a linear regression to predict the relative humidity based on all other variables.

7. Create a linear regression to predict the relative humidity based on all other variables using the SVD method. Copy the code here.
8. Running this on the data, identify the bias and the parameters.
9. What is the parameter associated with Barometric Pressure?
10. Predict the Relative Humidity with the following variables:
    * Temperature = $T$
    * Maximum Temperature = $X$
    * Minimum Temperature = $N$
    * Peak Wind Gust = $G$
    * Average Wind Speed = $W$
    * Wind Dir = 180
    * Barometric Pressure = $P$
    * Rain Total = 0
    * Maximum Dew Point Temperature = $D$
    * Minimum Dew Point Temperature = $E$