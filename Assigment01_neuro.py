import numpy as np
import matplotlib.pyplot as plt

'''
The Variable/Constants that are would be define the distributions Stimulus is generated using the normal/gaussian distribution and the response through binomial distribution function and parameters that are to be considers.

mu : mean of the nomal distribution
sigma : standard deviaton of then normal distribution
n : the number of trials 
p : probability of success or a spike
N : total number of samples drawn
w : the window size that can be varied which should be less than the number of samples drawn
'''
#mu, sigma = 0, 1
n = 1
#w = 300
#N = 1000

'''
::Enter the values::
'''
print('*** Enter the values of p, mu, sigma, w and N ***')
p = input('Enter the value of P(probability of Neuron firing) 0<P<1 :: ')
mu = input('Enter the value of mean (Stimulus distribution) :: ')
sigma = input('Enter the value of std (Stimulus distribution) :: ')
w = input('Enter the value of the sampling window :: ')
N = input('Enter the number of samples :: ')

'''
Genetare the stimulus sampled from the Normal distribution
c_s : contains an array of the values of thne histogram bins
b_s : the edges of the bins
'''
s = np.random.normal(mu, sigma, N)
c_s, b_s, i_s = plt.hist(s, 50, density=True)


plt.plot(range(0,1000,1),s,'b-')
plt.xlabel('Number of samples')
plt.ylabel('Value')
plt.title('Distribution of Stimulus')
plt.show()

'''
create the response by taking a binomial distribution that is generated randomly
Also, the response can be taken as a threshold function that fires at only a particular value
'''
#r = np.random.binomial(n, p, 1000)
r = [1 if i > 2.5 else 0 for i in s]

'''
returns a vector that contains 1's where the neuron fired and 0 at the places where it did not.
np.noozero(r) returns a tuple that contains all nonzero index values that contain 1's.
Tuple ==> converted to numpy array with np.asarray(idx)

**only taking the response after the first window length
'''
idx = np.nonzero(r)
idx = np.asarray(idx)
idx = idx[idx > w]

'''
create a variable 'f' to compute the spike triggered average
'''
f = np.zeros(w)

for i in range(len(idx)):
    f = f + s[idx[i]-w+1:idx[i]+1]

f = f/len(idx)

'''
Convolve the signal s with the STA filter to obtain the sf signal (smoothened) version of s
'''
sf = np.convolve(s,f)
c,b,i=plt.hist(sf, 50, density=True)
plt.plot(b,1/(sf.std() * np.sqrt(2*np.pi)) * np.exp(- (b - sf.mean())**2 / (2 * sf.std()**2)), linewidth=2, color='g')
plt.xlabel('Sampled interval bins')
plt.ylabel('Probability of occurance [P(Sf)]')
plt.title('Histogram of Sf')
plt.show()

'''
Obtain the distributions of Sf_r in regard to Sf and get a combined distribution plot
1. Plotting histogram of Sf
2. Plotting the distribution of Sf
3. Plotting the distribution of Sf_r
'''

sf_r = sf[idx]
ci,bi,ii=plt.hist(sf_r, 50, density=True)
plt.plot(b,1/(sf.std() * np.sqrt(2*np.pi)) * np.exp(- (b - sf.mean())**2 / (2 * sf.std()**2)), linewidth=2, color='g')
plt.plot(b,1/(sf_r.std() * np.sqrt(2*np.pi)) * np.exp(- (b - sf_r.mean())**2 / (2 * sf_r.std()**2)), linewidth=2, color='r')
plt.xlabel('Sampled interval bins')
plt.ylabel('Probability of occurance [P(Sf)] and [P(Sf_r)]')
plt.legend(['P(Sf)','P(Sf_r)','Hist'])
plt.title('Combined Histograms and plots of Sf and Sf_r')
plt.show()

'''
Obtain the non-linearity by taking a ratio of P(Sf_r) with P(Sf)
'''

p_sf_r = 1/(sf_r.std() * np.sqrt(2*np.pi)) * np.exp(- (b - sf_r.mean())**2 / (2 * sf_r.std()**2))
p_sf = 1/(sf.std() * np.sqrt(2*np.pi)) * np.exp(- (b - sf.mean())**2 / (2 * sf.std()**2))

plt.plot(b,p_sf_r/p_sf,linewidth=2,color='r')
plt.xlabel('Sampled interval bins')
plt.ylabel('P(Sf_r)/P(sf)')
plt.title('Ratio between the Probability plots')
plt.show()
