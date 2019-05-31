NEURON {
	POINT_PROCESS ExpSynSTDP
	RANGE tau, e, i, d, p, dtau, ptau, verbose, learning, LR, maxWeight, minWeight
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
	d = 0 <0,1>: depression factor (multiplicative to prevent < 0)
	p = 0 : potentiation factor (additive, non-saturating)
	dtau = 34 (ms) : 34 depression effectiveness time constant
	ptau = 17 (ms) : 17 Bi & Poo (1998, 2001)
	verbose = 0
	learning = 1
	LR = 0.0001
	maxWeight = 1
	minWeight = 0
}

ASSIGNED {
	v (mV)
	i (nA)
	tpost (ms)
}

STATE {
	g (uS)
}

INITIAL {
	g=0
	tpost = -1e9
	net_send(0, 1)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}

DERIVATIVE state {
	g' = -g/tau
}

NET_RECEIVE(w (uS), tpre (ms)) {
	INITIAL { tpre = -1e9 }
	if (flag == 0) { : presynaptic spike  (after last post so depress)
		g = g + w
		if(learning) {
			if (w>=minWeight){
				w = w-LR*d*exp((tpost - t)/dtau)
				if(w<=minWeight){
					w=minWeight
				}
				:w = w*LR*d*(1-(exp((tpost - t)/dtau)))
				if(verbose) {
					printf("dep: w=%g \t dw=%g \t dt=%g\n", w, -LR*d*exp((tpost - t)/dtau), tpost-t)
				}
			}
		}
		tpre = t
	}else if (flag == 2) { : postsynaptic spike
		tpost = t
		FOR_NETCONS(w1, tp) { : also can hide NET_RECEIVE args
        	if(learning) {
        		if (w1<=maxWeight){
	        		w1 = w1+LR*p*exp((tp - t)/ptau)
	        		if (w1>maxWeight){
	        			w1 = maxWeight
	        		}
	        		if(verbose) {
	        			printf("pot: w=%g \t dw=%g \t dt=%g\n", w1, (LR*p*exp((tp - t)/ptau)), t - tp)
	        		}
	        	}
        	}
		}
	} else { : flag == 1 from INITIAL block
		WATCH (v > -20) 2
	}
}
