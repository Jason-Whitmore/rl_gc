


public class DenseTanh extends Layer{

    private float[] sumVector;

    public DenseTanh(int numUnits, int inputSize){
        super();

        this.inputVector = new float[inputSize];
        this.outputVector = new float[numUnits];

        this.dObjectivedX = new float[inputSize];
        this.dObjectivedY = new float[numUnits];

        this.sumVector = new float[numUnits];

        //initialize weight and bias arrays
        this.parameters.add(new float[numUnits][inputSize]);
        this.parameters.add(new float[numUnits][1]);

        //initialize gradient arrays
        this.gradient.add(new float[numUnits][inputSize]);
        this.gradient.add(new float[numUnits][1]);

        this.populateParams(-0.000001f, 0.000001f);
    }

    public void forwardPass(){

        float[][] weights = this.parameters.get(0);
        float[][] bias = this.parameters.get(1);


        for(int i = 0; i < outputVector.length; i++){
            float sum = 0;

            sum += this.dot(weights[i], this.inputVector);
            sum += bias[i][0];

            this.sumVector[i] = sum;

            this.outputVector[i] = LSTM.tanh(sum);
        }
    }

    public void backwardPass(){
        float[][] weightsGradient = this.gradient.get(0);
        float[][] biasGradient = this.gradient.get(1);

        for(int i = 0; i < biasGradient.length; i++){
            biasGradient[i][0] = this.dObjectivedY[i];
        }

        for(int r = 0; r < weightsGradient.length; r++){
            for(int c = 0; c < weightsGradient[r].length; c++){
                weightsGradient[r][c] = this.inputVector[c] * LSTM.tanhPrime(this.sumVector[r]) * this.dObjectivedY[r];
            }
        }

        for(int i = 0; i < this.dObjectivedX.length; i++){
            float sum = 0;

            float[][] weights = this.parameters.get(0);
            for(int r = 0; r < weightsGradient.length; r++){
                sum += weights[r][i] * LSTM.tanhPrime(this.sumVector[r]) * this.dObjectivedY[r];
            }

            this.dObjectivedX[i] = sum;
        }
    }
}