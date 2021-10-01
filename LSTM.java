


public class LSTM extends Layer{

    public float[] currentCState;
    public float[] currentHState;

    public float[] nextCState;
    public float[] nextHState;


    private float[] dObjdO;
    private float[] dOdTanhC;
    private float[] dObjdC;
    private float[] dObjdF;
    private float[] dObjdI;
    private float[] dObjdCellOutput;

    private float[] outputGateVector;
    private float[] cellOutputVector;
    private float[] inputGateVector;

    private float[] fSum;
    private float[] iSum;
    private float[] oSum;

    //"Fancy C"
    private float[] cellActivationSum;

    public LSTM(int numUnits, int inputSize){
        super();

        //Create arrays for forward pass

        this.inputVector = new float[inputSize];
        this.outputVector = new float[numUnits];

        this.currentCState = new float[numUnits];
        this.nextCState = new float[numUnits];

        this.currentHState = new float[numUnits];
        this.nextHState = new float[numUnits];

        this.outputGateVector = new float[numUnits];
        this.cellOutputVector = new float[numUnits];
        this.inputGateVector = new float[numUnits];

        this.fSum = new float[numUnits];
        this.iSum = new float[numUnits];
        this.oSum = new float[numUnits];
        this.cellActivationSum = new float[numUnits];
        

        //Create backprop arrays
        this.dObjectivedX = new float[inputSize];
        this.dObjectivedY = new float[numUnits];


        //create parameter arrays
        //Forget gate (input weight, hidden weight, bias)
        this.parameters.add(new float[numUnits][inputSize]);
        this.parameters.add(new float[numUnits][numUnits]);
        this.parameters.add(new float[numUnits][1]);

        //update gate (input weight, hidden weight, bias)
        this.parameters.add(new float[numUnits][inputSize]);
        this.parameters.add(new float[numUnits][numUnits]);
        this.parameters.add(new float[numUnits][1]);

        //output gate (input weight, hidden weight, bias)
        this.parameters.add(new float[numUnits][inputSize]);
        this.parameters.add(new float[numUnits][numUnits]);
        this.parameters.add(new float[numUnits][1]);

        //cell activation (input weight, hidden weight, bias)
        this.parameters.add(new float[numUnits][inputSize]);
        this.parameters.add(new float[numUnits][numUnits]);
        this.parameters.add(new float[numUnits][1]);

        
        //Create gradient arrays
        for(int i = 0; i < this.parameters.size(); i++){
            float[][] array = new float[this.parameters.get(i).length][this.parameters.get(i)[0].length];
            this.gradient.add(array);
        }

        //Create vectors to help with gradient calculation
        this.dObjdO = new float[this.outputVector.length];
        this.dOdTanhC = new float[this.outputVector.length];
        this.dObjdC = new float[this.outputVector.length];
        this.dObjdF = new float[this.outputVector.length];
        this.dObjdCellOutput = new float[this.outputVector.length];
        this.dObjdI = new float[this.outputVector.length];


        this.populateParams(-0.1f, 0.1f);
    }



    public void forwardPass(){

        //Forget gate
        float[][] inputWeight = this.parameters.get(0);
        float[][] hiddenWeight = this.parameters.get(1);
        float[][] bias = this.parameters.get(2);

        for(int i = 0; i < this.fSum.length; i++){
            float sum = 0;

            sum += this.dot(inputWeight[i], this.inputVector);
            sum += this.dot(hiddenWeight[i], this.currentHState);
            sum += bias[i][0];

            this.fSum[i] = sum;
        }


        //Input gate
        inputWeight = this.parameters.get(3);
        hiddenWeight = this.parameters.get(4);
        bias = this.parameters.get(5);

        for(int i = 0; i < this.iSum.length; i++){
            float sum = 0;

            sum += this.dot(inputWeight[i], this.inputVector);
            sum += this.dot(hiddenWeight[i], this.currentHState);
            sum += bias[i][0];

            this.iSum[i] = sum;
        }

        //Output gate
        inputWeight = this.parameters.get(6);
        hiddenWeight = this.parameters.get(7);
        bias = this.parameters.get(8);

        for(int i = 0; i < this.oSum.length; i++){
            float sum = 0;

            sum += this.dot(inputWeight[i], this.inputVector);
            sum += this.dot(hiddenWeight[i], this.currentHState);
            sum += bias[i][0];

            this.oSum[i] = sum;
        }

        //Cell activation
        inputWeight = this.parameters.get(9);
        hiddenWeight = this.parameters.get(10);
        bias = this.parameters.get(11);

        for(int i = 0; i < this.cellActivationSum.length; i++){
            float sum = 0;

            sum += this.dot(inputWeight[i], this.inputVector);
            sum += this.dot(hiddenWeight[i], this.currentHState);
            sum += bias[i][0];

            this.cellActivationSum[i] = sum;
        }


        //Cell state, hidden state, output vector
        for(int i = 0; i < this.currentCState.length; i++){
            float f = (float)Math.tanh((double)this.fSum[i]);
            float input = sigmoid(this.iSum[i]);
            float cActivation = LSTM.tanh(this.cellActivationSum[i]);

            this.nextCState[i] = f * this.currentCState[i] + input * cActivation;

            this.nextHState[i] = this.sigmoid(this.oSum[i]) * LSTM.tanh(this.currentCState[i]);

            this.outputVector[i] = this.currentHState[i];
        }



    }


    public void backwardPass(){
        //Assume dObjdY is populated correctly. Calculate dObjdO
        for(int i = 0; i < this.dObjdO.length; i++){
            this.dObjdO[i] = this.dObjectivedY[i] * (float)Math.tanh((double)this.currentCState[i]);
        }

        //Calculate dObjdTanhC
        for(int i = 0; i < this.dOdTanhC.length; i++){
            this.dOdTanhC[i] = this.dObjectivedY[i] * this.outputGateVector[i];
        }

        //Calculate dObjdC
        for(int i = 0; i < this.dObjdC.length; i++){
            this.dObjdC[i] = this.dOdTanhC[i] * this.tanhPrime(this.currentCState[i]);
        }

        //Calculate dObjdF
        for(int i = 0; i < this.dObjdF.length; i++){
            this.dObjdF[i] = this.dObjdC[i] * this.currentCState[i];
        }

        //Calculate dObjdI
        for(int i = 0; i < this.dObjdI.length; i++){
            this.dObjdI[i] = this.dObjdC[i] * this.cellOutputVector[i];
        }

        //Calculate dObjdCellOutput
        for(int i = 0; i < this.dObjdCellOutput.length; i++){
            this.dObjdCellOutput[i] = this.dObjdC[i] * this.inputGateVector[i];
        }

        //All helper arrays calculated. Calculate gradient wrt params
        //Forget gate (input weight, hidden weight, bias)
        float[][] inputWeightGradient = this.gradient.get(0);

        for(int r = 0; r < inputWeightGradient.length; r++){
            for(int c = 0; c < inputWeightGradient[r].length; c++){
                inputWeightGradient[r][c] = this.dObjdF[r] * this.sigmoidPrime(this.fSum[r]) * this.inputVector[c];
            }
        }



        float[][] hiddenWeightGradient = this.gradient.get(1);

        for(int r = 0; r < hiddenWeightGradient.length; r++){
            for(int c = 0; c < hiddenWeightGradient[r].length; c++){
                hiddenWeightGradient[r][c] = this.dObjdF[r] * this.sigmoidPrime(this.fSum[r]) * this.currentHState[c];
            }
        }



        float[][] biasGradient = this.gradient.get(2);

        for(int r = 0; r < biasGradient.length; r++){
            biasGradient[r][0] = this.dObjdF[r] * this.sigmoidPrime(this.fSum[r]);
        }


        
        //Input gate (input weight, hidden weight, bias)
        inputWeightGradient = this.gradient.get(3);

        for(int r = 0; r < inputWeightGradient.length; r++){
            for(int c = 0; c < inputWeightGradient[r].length; c++){
                inputWeightGradient[r][c] = this.dObjdI[r] * this.sigmoidPrime(this.iSum[r]) * this.inputVector[c];
            }
        }


        hiddenWeightGradient = this.gradient.get(4);

        for(int r = 0; r < hiddenWeightGradient.length; r++){
            for(int c = 0; c < hiddenWeightGradient[r].length; c++){
                hiddenWeightGradient[r][c] = this.dObjdI[r] * this.sigmoidPrime(this.iSum[r]) * this.currentHState[c];
            }
        }


        biasGradient = this.gradient.get(5);

        for(int r = 0; r < biasGradient.length; r++){
            biasGradient[r][0] = this.dObjdF[r] * this.sigmoidPrime(this.iSum[r]);
        }

        //Output gate (input weight, hidden weight, bias)

        inputWeightGradient = this.gradient.get(6);

        for(int r = 0; r < inputWeightGradient.length; r++){
            for(int c = 0; c < inputWeightGradient[r].length; c++){
                inputWeightGradient[r][c] = this.dObjdO[r] * this.sigmoidPrime(this.oSum[r]) * this.inputVector[c];
            }
        }


        hiddenWeightGradient = this.gradient.get(7);

        for(int r = 0; r < hiddenWeightGradient.length; r++){
            for(int c = 0; c < hiddenWeightGradient[r].length; c++){
                hiddenWeightGradient[r][c] = this.dObjdO[r] * this.sigmoidPrime(this.oSum[r]) * this.currentHState[c];
            }
        }


        biasGradient = this.gradient.get(8);

        for(int r = 0; r < biasGradient.length; r++){
            biasGradient[r][0] = this.dObjdO[r] * this.sigmoidPrime(this.oSum[r]);
        }


        //Cell activation gate (input weight, hidden weight, bias)

        inputWeightGradient = this.gradient.get(9);

        for(int r = 0; r < inputWeightGradient.length; r++){
            for(int c = 0; c < inputWeightGradient[r].length; c++){
                inputWeightGradient[r][c] = this.dObjdCellOutput[r] * this.tanhPrime(this.cellActivationSum[r]) * this.inputVector[c];
            }
        }


        hiddenWeightGradient = this.gradient.get(10);

        for(int r = 0; r < hiddenWeightGradient.length; r++){
            for(int c = 0; c < hiddenWeightGradient[r].length; c++){
                hiddenWeightGradient[r][c] = this.dObjdCellOutput[r] * this.tanhPrime(this.cellActivationSum[r]) * this.currentHState[c];
            }
        }


        biasGradient = this.gradient.get(11);

        for(int r = 0; r < biasGradient.length; r++){
            biasGradient[r][0] = this.dObjdCellOutput[r] * this.tanhPrime(this.cellActivationSum[r]);
        }

        //Populate the dObjdX array

        for(int i = 0; i < this.dObjectivedX.length; i++){
            float sum = 0;

            //Visit each of the gates an use the weight array to update the gradient of the ith component of the input array
            float[][] weights = null;

            for(int j = 0; j < this.gradient.size(); j+=3){
                weights = this.gradient.get(j);
                for(int r = 0; r < weights.length; r++){
                    sum += weights[r][i];
                }
            }

            this.dObjectivedX[i] = sum;
        }


    }

    public void updateState(){

        GCAgent.copyArrayContents(this.nextCState, this.currentCState);
        GCAgent.copyArrayContents(this.nextHState, this.currentHState);
    }

    public static float tanh(float x){
        return (float)Math.tanh((double)x);
    }

    public static float tanhPrime(float x){
        return (float)(1.0 - ((Math.tanh((double)x)) * (Math.tanh((double)x))));
    }

    private float sigmoid(float x){
        double s = 1.0 / (1 + Math.exp((double)-x));

        return (float)s;
    }

    private float sigmoidPrime(float x){
        return this.sigmoid(x) * (1 - this.sigmoid(x));
    }

}