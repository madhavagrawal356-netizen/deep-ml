import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	for i in range(len(consumption_data)):
		consumption_data[i]-=10*math.sin(2*PI*(i+1)/10)
	days = [i for i in range(1,len(consumption_data)+1)]
	mean_x = sum(days)/len(days)
	mean_y = sum(consumption_data)/len(consumption_data)
	s_xx = sum((xi-mean_x)**2 for xi in days)
	s_xy = sum((xi-mean_x)*(yi-mean_y) for xi,yi in zip(days,consumption_data))
	slope = s_xy/s_xx
	intercept = mean_y-slope*mean_x
	day_cons =slope*15+intercept+math.sin(2*PI*15/10)
	day_cons*=1.05
	return int(math.ceil(day_cons))

