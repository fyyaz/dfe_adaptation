# 2020/08/02
# simulation of sign sign lms algorithm for dfe adaptation

import numpy as np 
from matplotlib import pyplot as plt 

DEBUG = False

#number of taps used for channel ISI (including the main tap)
CHANNEL_TAPS = 4 

#number of taps used by DFE to cancel ISI + main tap, there are DFE_TAPS -1 taps available to cancel postcursor ISI
DFE_TAPS = 8

#width of data to be serialized/deserialized by tx/rx
DATA_WIDTH = 7

#channel impulse response
H = [1.0, 0.9, -0.5, 0.3, -0.2, 0, 0]

#adaptation gain for each DFE coeffecient
mu = DFE_TAPS * [0.01]

#dfe tap coeffecients
c = [0] * DFE_TAPS # DFE tap coeffecients

#previous bits sent by TX
prev_d_tx = [0] * CHANNEL_TAPS # previous bits sent by tx

#previous bits recieved by rx
prev_d_rx = [0] * DFE_TAPS

#to see coeffecient changes
dfe = [[] for _ in range(DFE_TAPS)]

#disable tap n from adapting if adapt[n] = 0
adapt = [1] * DFE_TAPS;

tx = [] #tx output
rx = [] #post dfe rx output

## generates a prbs7 sequence using start as the input seed 
## output data width is 7 bits
def prbs7(start):
	#characteristic polynomial: x7 + x6 + 1 (1 based)
	#data width: 7 bits
	newbit = ((start >> 6) ^ (start >> 5)) & 0x01
	start = (start << 1 | newbit) & 0x7f

	return start

## checks if prbs7() has 2^7-1 as period
def test_prbs7():
	start = 5;
	count = 0;
	curr = start;

	while True:
		print(format(curr, '07b'))
		curr = prbs7(curr)
		count = count + 1
		#print(count)
		if curr == start:
			break

	print('prbs period is: %d, expected: %d' % (count, 2**7-1))

## applies channel reponse to sent bit
def send(txbit):
	# shift all the data to the next index
	for i in range(CHANNEL_TAPS-1, 0, -1):
		prev_d_tx[i] = prev_d_tx[i-1];

	main_sign = txbit * 2 - 1 #shift txbit from [0,1] to [-1, +1]
	prev_d_tx[0] = main_sign

	rxbit = 0
	for i in range(CHANNEL_TAPS):
		rxbit = rxbit + prev_d_tx[i] * H[i]

	return rxbit

def reset_sys():
	global c, prev_d_tx, prev_d_rx, dfe
	c = [0] * DFE_TAPS
	prev_d_tx = [0] * CHANNEL_TAPS
	prev_d_rx = [0] * DFE_TAPS
	dfe = [[] for _ in range(DFE_TAPS)]

## input: array of dfe coeffecients to set dfe
def set(f):
	for i in range(DFE_TAPS):
		if i < len(f):
			c[i] = f[i]
		else:
			c[i] = 0
def dfe_adapt_graph():
	for i in range(DFE_TAPS):
		ax1 = plt.subplot(4,2,i+1)
		ax1.set_title('F%d' % i)
		plt.plot(dfe[i])

	plt.show()
def sim(num_words=20, show=False, reset=False, seed=10, freeze_dfe=False, speculate_d0=True):
	global tx, rx

	if reset: #resets the systems memory
		reset_sys()

	txdata = prbs7(seed)
	tx = [] #tx output
	pre_rx = [] #rx input
	rx = [] #post dfe rx output
	
	bitcount = 0
	count = 0
	if speculate_d0:
		d0_predict = 1
	else: d0_predict = 0

	while num_words != count:
		## serialize 7 bit txdata and send it 
		txdata_shift = txdata

		for i in range(DATA_WIDTH):
			if DEBUG:
				print('bit #: ', bitcount)

			txbit = txdata_shift & 0x1 
			rxbit = send(txbit)

			if DEBUG:
				print('tx sent data: ', 2*txbit-1)
				print('data after ch:, ', rxbit)

			rxout = recv(rxbit, 0)
			rxd = rxout[0]
			rxe = rxout[1]
			txdata_shift = txdata_shift >> 1
			tx.append(2*txbit-1)
			pre_rx.append(rxbit) 
			rx.append(rxd)

			for i in range(DFE_TAPS):
				dfe[i].append(c[i])

			bitcount = bitcount + 1
			if bitcount % 3 == 0:
				d0_predict = -d0_predict

		count = count + 1
		txdata = prbs7(txdata)
	if (show):
		plt.figure(1)
		plt.subplot(211)
		plt.stem(tx)
		plt.subplot(212)
		plt.stem(rx)
		plt.show()

def sign(n):
	if n > 0:
		return 1
	else:
		return -1

def recv(rxin, d0_predict=0, freeze_dfe=False):

	#move all previous data in dfe delay line forward
	for i in range(DFE_TAPS-1, 0, -1):
		prev_d_rx[i] = prev_d_rx[i-1];


	#create the data and error signals
	rxd = rxin
	for i in range(1, DFE_TAPS):
		rxd = rxd - prev_d_rx[i] * c[i]

	if (d0_predict == 0):
		prev_d_rx[0] = sign(rxd); #note in real dfe need to take turns guessing prev_d_rx[0]
	elif (d0_predict == 1):
		prev_d_rx[0] = 1
		#print('adapting if 1')
	else:
		prev_d_rx[0] = -1
		#print('adapting if -1')

	rxe = rxd - prev_d_rx[0] * c[0];

	#adapt dfe taps
	#import pdb;pdb.set_trace();

	if DEBUG:
		print ('rx data: ', rxd)
		print ('rx error', rxe)
		print('adapt: ', d0_predict)

	if sign(rxd) == prev_d_rx[0] and not freeze_dfe:
		for i in range(DFE_TAPS):
			c[i] = c[i] + mu[i]*sign(rxe)*sign(prev_d_rx[i])*adapt[i]
		#print('change')
	else:
		#print('no change')
		pass

	rxd_out = (sign(rxd))
	rxe_out = (sign(rxe))

	prev_d_rx[0] = sign(rxd)

	return [rxd_out, rxe_out]

def channel_response():
	plt.stem(H)
	plt.show()
	pass

def dfe_response():
	plt.stem(c)
	plt.show();
	pass

def get_ber():
	errors = 0
	bits = len(tx) 
	for i in range(len(tx)):
		errors = errors + 1 * (tx[i] != rx[i])

	return errors/bits
