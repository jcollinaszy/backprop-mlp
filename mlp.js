const Nin = 2;				// number of inputs
const Nh1 = 2;				// number of hidden units
const Nou = 1;				// number of outputs
const Gamma = 0.2;			// learning rate
const Epochs = 40000;		// number of training cycles

const Nx = 1+Nin+Nh1+Nou; 	// number of units
const IN1 = 1;				// 1st input
const INn = Nin;			// last (n-th) input
const H11 = Nin+1;			// 1st hidden
const H1n = Nin+Nh1;		// last hidden
const OU1 = H1n+1;			// 1st output
const OUn = H1n+Nou;		// last output

var xor = [[0, 0, 0],[0, 1, 1],[1, 0, 1],[1, 1, 0]];

// main structure
class Ann {     
	constructor(Nx, Nou) {         
		this.x = new Array(Nx);         
		this.y = new Array(Nx);         
		this.delta = new Array(Nx);                  
		this.w = new Array(Nx);         
		for (var i = 0; i < Nx; ++i) {             
			this.w[i] = new Array(Nx);         
		}                  
		this.dv = new Array(Nou);     
	} 
}

// activation function
function af(x) {
	return (1.0 / (1.0 + Math.exp((-1.0) * x)));
}

// first derivation of activation function
function df(x) {
	return (Math.exp((-1.0) * x) / ((1.0 + Math.exp((-1.0) * x)) * (1.0 + Math.exp((-1.0) * x))));
}

// random weights initialization
function randomInit(ann, min, max) {
	ann.y[0] = -1.0;
	for (var i = 0; i < Nx; i++) {
		for (var j = 0; j < Nx; j++) {
			ann.w[i][j] = Math.random() * (max - min) + min;
		}
	}
}

// run a single layer
function layerRun(ann, i1, inn, j1, jn) {	
	for (var i = i1; i <= inn; i++) {
		ann.x[i] = ann.w[i][0] * ann.y[0];		
		for (var j = j1; j <= jn; j++) {
			ann.x[i] += ann.w[i][j] * ann.y[j];
		}
		ann.y[i] = af(ann.x[i]);
	}
}

// update weights on single layer
function layerUpdate(ann, i1, inn, j1, jn, gamma) {	
	for (var i = i1; i <= inn; i++) {					
		ann.w[i][0] += gamma * ann.delta[i] * ann.y[0];			
		for (var j = j1; j <= jn; j++) {
			ann.w[i][j] += gamma * ann.delta[i] * ann.y[j]; 
		}			
	}
}		

// run the whole network
function mlpRun(ann) {					
	layerRun(ann, H11, H1n, IN1, INn);
	layerRun(ann, OU1, OUn, H11, H1n);
}

// backpropagation training
function vanillaBp(ann, gamma) {		
	mlpRun(ann);			
	for (var i = OU1; i <= OUn; i++) {
		ann.delta[i] = (ann.dv[i - OU1] - ann.y[i]) * df(ann.x[i]);
	}
	for (var i = H11; i <= H1n; i++) {
		var S = 0.0;
		for (var h = OU1; h <= OUn; h++) {
			S += ann.delta[h] * ann.w[h][i];
		}
		ann.delta[i] = S * df(ann.x[i]);
	}							
	layerUpdate(ann, OU1, OUn, H11, H1n, gamma);					
	layerUpdate(ann, H11, H1n, IN1, INn, gamma); 
}	  

var ann = new Ann(Nx, Nou);
randomInit(ann, -0.1, 0.1);

for(var epoch = 0; epoch <= Epochs; epoch++) {			
    for(var p = 0; p < xor.length; p++) {						
		ann.y[IN1] = xor[p][0];							
		ann.y[IN1+1] = xor[p][1];								
		ann.dv[0] = xor[p][2];								
		vanillaBp(ann, Gamma);				
		if (epoch % 5000==0) {								
			if(p == 0 && epoch != 0) console.log("");			
			mlpRun(ann);									
			var J = Math.abs(ann.dv[0] - ann.y[OU1]);				
			console.log(epoch, " - error - ", J);
		}
	}
}