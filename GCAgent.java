import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.Arrays;

public class GCAgent{

    private float policyLR;

    private float valueLR;


    private float updateStepSize;

    private int minUpdateInterval;

    private float confidenceStopThreshold;


    private float discountFactor;


    private int observationSize;

    private int stateSize;

    private long prevObsTime;

    private boolean firstTimestep;


    private ArrayList<Layer> policyNetwork;

    private ArrayList<Layer> updateNetwork;

    private ArrayList<Layer> valueNetwork;

    private float[] obs;

    private float probAction;

    private int prev_action;

    private float[] state;

    private float bestMeanReward;
    private ArrayList<float[][]> bestUpdateParams;

    //Stopping conditions/debug statistics
    private MovingAverage meanConfidence;
    private MovingAverage meanReward;


    public GCAgent(){
        this.stateSize = 200;

        this.valueLR = 0.001f;
        this.policyLR = 0.0001f;
        this.updateStepSize = 0.001f;
        this.confidenceStopThreshold = 0.98f;
        this.discountFactor = 0.9999f;
        this.minUpdateInterval = 10000;

        //Create the functions
        this.createPolicyNetwork(64);
        this.createValueNetwork(64);
        this.createUpdateNetwork(64);

        //Other fields
        this.firstTimestep = true;
        this.bestMeanReward = Float.NEGATIVE_INFINITY;
        this.meanReward = new MovingAverage(this.minUpdateInterval);
        this.meanConfidence = new MovingAverage(this.minUpdateInterval);
    }

    public GCAgent(int stateSize, int hiddenLayerSizePolicy, int hiddenLayerSizeValue, int hiddenLayerSizeUpdate, float valueLearningRate, float policyLearningRate, float updateStepSize, float confidenceStopThreshold, float discountFactor, int minUpdateInterval){

        this.stateSize = stateSize;

        this.valueLR = valueLearningRate;
        this.policyLR = policyLearningRate;
        this.updateStepSize = updateStepSize;
        this.confidenceStopThreshold = confidenceStopThreshold;
        this.discountFactor = discountFactor;
        this.minUpdateInterval = minUpdateInterval;

        //Create the functions
        this.createPolicyNetwork(hiddenLayerSizePolicy);
        this.createValueNetwork(hiddenLayerSizeValue);
        this.createUpdateNetwork(hiddenLayerSizeUpdate);

        //Other fields
        this.firstTimestep = true;
        this.bestMeanReward = Float.NEGATIVE_INFINITY;
        this.meanReward = new MovingAverage(this.minUpdateInterval);
        this.meanConfidence = new MovingAverage(this.minUpdateInterval);
    }

    /**
     * Invokes the agent, which does the TD error update, updates the internal agent/value states and parameters and performs the action.
     */
    public void invoke(){
        
        //If this is the first timestep, do not perform the TD error update. Instead, just select an action and invoke.
        if(this.firstTimestep){

            //Get the current observation
            this.obs = this.getObservation();

            //Run input update input vector through network to get the state
            this.state = this.updateFunctionPredictFirstTimestep(obs);

            //Select action from policy
            float[] probVector = this.policyFunctionPredict(state);
            this.prev_action = this.selectAction(probVector);

            if(prev_action == 1){
                System.gc();
            }

            //Record the probability of the action being selected at the current state
            this.probAction = probVector[this.prev_action];

            //Record the "confidence" level (maximum probability out of policy distribution)
            this.meanConfidence.addSample(Math.max(this.probAction, 1 - this.probAction));

            this.firstTimestep = false;
        } else {

            //Get the next observation
            float[] nextObs = this.getObservation();

            //Get the new reward
            long currentObsTime = System.currentTimeMillis();
            long delta = currentObsTime - this.prevObsTime;
            float deltaSeconds = ((float)delta) / 1000f;
            this.prevObsTime = currentObsTime;

            float reward = -(float)deltaSeconds;
            this.meanReward.addSample(reward);

            //Get next state
            float[] nextState = this.updateFunctionPredict(this.state, this.prev_action, nextObs);

            //Get values
            float nextValue = this.valueFunctionPredict(nextState);
            float value = this.valueFunctionPredict(this.state);

            //Calculate the TD error
            float tdError = reward + (this.discountFactor * nextValue) - value;



            //Adjust value function
            //Backprop data should already be stored in the value function object since the last call to predict was on the "current" inputs
            float[] dObjdY = new float[1];
            dObjdY[0] = 1;
            this.neuralNetworkGradient(this.valueNetwork, dObjdY);
            this.applyGradients(this.valueNetwork, tdError * this.valueLR);

            //Adjust the policy function
            //Run the predict function to populate the backprop data in the function object
            float[] policyOutput = this.neuralNetworkPredict(this.policyNetwork, this.state);
            dObjdY = this.softmaxdObjdX(policyOutput, this.prev_action);
            this.neuralNetworkGradient(this.policyNetwork, dObjdY);
            this.applyGradients(this.policyNetwork, tdError * this.policyLR);
            
            
            //Select the next action
            float[] probVector = this.policyFunctionPredict(nextState);
            this.prev_action = this.selectAction(probVector);

            if(this.prev_action == 1){
                System.gc();
            }

            //Set the "current" observation, action probability, and state fields to the past fields
            this.obs = nextObs;
            this.probAction = probVector[this.prev_action];
            this.state = nextState;

            this.meanConfidence.addSample(Math.max(this.probAction, 1 - this.probAction));

        }


            

        //Determine if update function should be adjusted
        if(this.meanReward.getNumSamples() > this.minUpdateInterval && this.meanConfidence.getMean() > this.confidenceStopThreshold){
            if(this.meanReward.getMean() > this.bestMeanReward){
                //Current update params are better than previous, set variables
                this.bestMeanReward = this.meanReward.getMean();
                this.bestUpdateParams = GCAgent.copyParams(this.updateNetwork);
            }
            
            System.out.println("Best mean reward: " + this.bestMeanReward);
            
            //Create new update function
            this.meanReward.reset();
            this.setParameters(this.updateNetwork, this.bestUpdateParams);
            this.randomOffsetParams(this.updateNetwork, this.updateStepSize);
            this.firstTimestep = true;
        }

    }



    private void createPolicyNetwork(int hiddenLayerSize){
        this.policyNetwork = new ArrayList<Layer>();

        //Use 2 hidden layers
        this.policyNetwork.add(new DenseTanh(hiddenLayerSize, this.stateSize));
        this.policyNetwork.add(new DenseTanh(hiddenLayerSize, hiddenLayerSize));

        //Linear dense layer for the output
        this.policyNetwork.add(new DenseLinear(2, hiddenLayerSize));

        //set the last layer weights to equal zero so that the starting policy is uniform random
        DenseLinear lastLayer = (DenseLinear)this.policyNetwork.get(this.policyNetwork.size() - 1);
        float[][] weight = lastLayer.parameters.get(0);
        for(int r = 0; r < weight.length; r++){
            for(int c = 0; c < weight[r].length; c++){
                weight[r][c] = 0f;
            }
        }
    }

    private void createValueNetwork(int hiddenSize){
        this.valueNetwork = new ArrayList<Layer>();

        this.valueNetwork.add(new DenseTanh(hiddenSize, this.stateSize));
        this.valueNetwork.add(new DenseTanh(hiddenSize, hiddenSize));
        this.valueNetwork.add(new DenseLinear(1, hiddenSize));
    }

    private void createUpdateNetwork(int hiddenUnitSize){
        this.updateNetwork = new ArrayList<Layer>();

        this.updateNetwork.add(new DenseTanh(hiddenUnitSize, this.stateSize + this.observationSize + 2));
        this.updateNetwork.add(new DenseTanh(hiddenUnitSize, hiddenUnitSize));

        this.updateNetwork.add(new DenseTanh(this.stateSize, hiddenUnitSize));
    }

    private float[] updateFunctionPredict(float[] prevState, int prevAction, float[] obs){
        float[] inputVector = this.getUpdateInputVector(prevState, prevAction, obs);

        return this.neuralNetworkPredict(this.updateNetwork, inputVector);
    }

    private float[] updateFunctionPredictFirstTimestep(float[] firstObs){
        float[] inputVector = new float[this.stateSize + 2 + firstObs.length];

        for(int i = 0; i < this.stateSize + 2; i++){
            inputVector[i] = -1;
        }

        int offset = this.stateSize + 2;
        for(int i = 0; i < firstObs.length; i++){
            inputVector[i + offset] = firstObs[i];
        }

        return this.neuralNetworkPredict(this.updateNetwork, inputVector);
    }

    private float valueFunctionPredict(float[] state){
        return this.neuralNetworkPredict(this.valueNetwork, state)[0];
    }

    private float[] policyFunctionPredict(float[] state){
        return this.softmax(this.neuralNetworkPredict(this.policyNetwork, state));
    }

    private float[] getUpdateInputVector(float[] prevState, int prevAction, float[] obs){
        float[] inputVector = new float[prevState.length + 2 + obs.length];

        //Copy over prevState
        for(int i = 0; i < prevState.length; i++){
            inputVector[i] = prevState[i];
        }

        //One hot encode the previous action
        inputVector[prevState.length + prevAction] = 1;

        for(int i = 0; i < obs.length; i++){
            inputVector[prevState.length + 2 + i] = obs[i];
        }

        return inputVector;
    }

    private float[] getStartingUpdateInputVector(float[] obs){
        float[] inputVector = new float[this.stateSize + 2 + obs.length];

        for(int i = 0; i < obs.length; i++){
            inputVector[this.stateSize + 2 + i] = obs[i];
        }

        return inputVector;
    }



    private float[] neuralNetworkPredict(ArrayList<Layer> neuralNetwork, float[] inputVector){

        GCAgent.copyArrayContents(inputVector, neuralNetwork.get(0).inputVector);
        //neuralNetwork.get(0).inputVector = inputVector;
        neuralNetwork.get(0).forwardPass();

        for(int i = 1; i < neuralNetwork.size(); i++){
            GCAgent.copyArrayContents(neuralNetwork.get(i - 1).outputVector, neuralNetwork.get(i).inputVector);

            neuralNetwork.get(i).forwardPass();
        }

        return neuralNetwork.get(neuralNetwork.size() - 1).outputVector;
    }

    private void neuralNetworkGradient(ArrayList<Layer> neuralNetwork, float[] dObjdY){
        Layer last = neuralNetwork.get(neuralNetwork.size() - 1);
        GCAgent.copyArrayContents(dObjdY, last.dObjectivedY);
        last.backwardPass();

        for(int i = neuralNetwork.size() - 2; i >= 0; i--){
            GCAgent.copyArrayContents(neuralNetwork.get(i + 1).dObjectivedX, neuralNetwork.get(i).dObjectivedY);

            neuralNetwork.get(i).backwardPass();
        }
    }

    private int getTotalEntryCount(ArrayList<float[]> list){
        int count = 0;

        for(int i = 0; i < list.size(); i++){
            count += list.get(i).length;
        }

        return count;
    }

    private void applyGradients(ArrayList<Layer> neuralNetwork, float scalar){

        for(int i = 0; i < neuralNetwork.size(); i++){
            Layer l = neuralNetwork.get(i);

            l.applyGradient(scalar);
        }

    }

    public static void copyArrayContents(float[] src, float[] dest){
        for(int i = 0; i < src.length; i++){
            dest[i] = src[i];
        }
    }


    public float[] softmax(float[] x){
        float[] y = new float[x.length];

        float sum = 0;

        for(int i = 0; i < x.length; i++){
            sum += (float)Math.exp((double)x[i]);
        }

        for(int i = 0; i < y.length; i++){
            y[i] = (float)Math.exp((double)x[i]) / sum;
        }

        return y;
    }


    public float[] softmaxdObjdX(float[] x, int index){
        float[] dObjdX = new float[x.length];

        float sum = 0;

        for(int i = 0; i < x.length; i++){
            sum += (float)Math.exp((double)x[i]);
        }

        //dObjdX for the main index

        float indexExp = (float)Math.exp((double)x[index]);
        sum -= indexExp;

        dObjdX[index] = (sum * indexExp) / ((sum + indexExp) * (sum + indexExp));

        sum += indexExp;

        for(int i = 0; i < x.length; i++){
            if(i != index){
                sum -= (float)Math.exp((double)x[i]);

                dObjdX[i] = -((indexExp * (float)Math.exp((double)x[i])) / (((float)Math.exp((double)x[i]) + sum) * ((float)Math.exp((double)x[i]) + sum)));

                sum += (float)Math.exp((double)x[i]);
            }
        }

        return dObjdX;
    }


    private int selectAction(float[] probVector){
        float r = (float)Math.random();

        for(int i = 0; i < probVector.length; i++){

            r -= probVector[i];

            if(r <= 0){
                return i;
            }
        }

        return 0;
    }


    private float[] getObservation(){
        float[] observation = new float[this.observationSize];

        Runtime r = Runtime.getRuntime();

        long totalMemory = r.totalMemory();
        long freeMemory = r.freeMemory();
        long maxMemory = r.maxMemory();

        //Index 0: Amount of memory used (in GB)
        observation[0] = (float)((totalMemory) / (1024.0 * 1024 * 1024));

        //Index 1: Max memory (in GB)
        observation[1] = (float)((maxMemory) / (1024.0 * 1024 * 1024));

        //Index 2: Proportion of memory used
        observation[2] = (float)(1.0 - (((double)(freeMemory)) / totalMemory));

        //Index 3: Available processors (divided by 128 to keep value small)
        observation[3] = r.availableProcessors() / 128f;

        //Next indexes are based on the mxbean information.
        MemoryMXBean mem = ManagementFactory.getMemoryMXBean();

        //Look at heap information
        MemoryUsage heap = mem.getHeapMemoryUsage();

        //Index 4: initial heap size
        observation[4] = (float)((heap.getInit()) / (1024.0 * 1024 * 1024));

        //Index 5: heap committed memory
        observation[5] = (float)((heap.getCommitted()) / (1024.0 * 1024 * 1024));

        //Index 6: heap used memory
        observation[6] = (float)((heap.getUsed()) / (1024.0 * 1024 * 1024));

        //Index 7: heap max memory
        observation[7] = (float)((heap.getMax()) / (1024.0 * 1024 * 1024));


        //Look at non-heap information (don't know if it will be useful to the agent)
        MemoryUsage nonHeap = mem.getNonHeapMemoryUsage();

        //Index 8: initial heap size
        observation[8] = (float)((nonHeap.getInit()) / (1024.0 * 1024 * 1024));

        //Index 9: heap committed memory
        observation[9] = (float)((nonHeap.getCommitted()) / (1024.0 * 1024 * 1024));

        //Index 10: heap used memory
        observation[10] = (float)((nonHeap.getUsed()) / (1024.0 * 1024 * 1024));

        //Index 11: heap max memory
        observation[11] = (float)((nonHeap.getMax()) / (1024.0 * 1024 * 1024));
        

        return observation;
    }

    public static void printArray(float[] array){
        System.out.print("[");

        for(int i = 0; i < array.length - 1; i++){
            System.out.print(array[i] + ", ");
        }

        System.out.print(array[array.length - 1] + "]\n");
    }

    private static ArrayList<float[][]> copyParams(ArrayList<Layer> network){
        ArrayList<float[][]> params = new ArrayList<float[][]>();

        for(int i = 0; i < network.size(); i++){
            for(int j = 0; j < network.get(i).parameters.size(); j++){
                params.add(GCAgent.copy(network.get(i).parameters.get(j)));
            }
        }

        return params;
    }

    private static float[][] copy(float[][] matrix){
        float[][] result = new float[matrix.length][matrix[0].length];

        for(int r = 0; r < result.length; r++){
            for(int c = 0; c < result[0].length; c++){
                result[r][c] = matrix[r][c];
            }
        }

        return result;
    }

    private void setParameters(ArrayList<Layer> network, ArrayList<float[][]> parameters){
        int k = 0;
        for(int i = 0; i < network.size(); i++){
            Layer l = network.get(i);

            for(int j = 0; j < l.parameters.size(); i++){
                l.parameters.set(j, GCAgent.copy(parameters.get(k)));
                k++;
            }
        }
    }

    private void randomOffsetParams(ArrayList<Layer> network, float maxOffset){
        for(int i = 0; i < network.size(); i++){
            Layer l = network.get(i);

            for(int j = 0; j < l.parameters.size(); j++){
                float[][] matrix = l.parameters.get(j);

                Utility.populateRandom(matrix, -maxOffset, maxOffset);
            }
        }
    }

}