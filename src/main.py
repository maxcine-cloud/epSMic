# -*- coding: utf-8 -*-

from arg import *
import test
import train

if "__main__" == __name__:
    
	args=get_args()
	if args.dataType == 'test':
		test.test1()
	else:
		train.train1()