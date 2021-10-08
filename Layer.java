import java.util.ArrayList;

public abstract class Layer{

    protected float[] dObjectivedY;
    protected float[] dObjectivedX;

    protected float[] inputVector;
    protected float[] outputVector;

    protected ArrayList<float[][]> parameters;
    protected ArrayList<float[][]> gradient;

    public Layer(){
        parameters = new ArrayList<float[][]>();
        gradient = new ArrayList<float[][]>();
    }


    public abstract void forwardPass();

    /**
     * Goal: Given dObjectivedY, calculate derivative data into gradient arraylist and populate dObjectivedX array.
     */
    public abstract void backwardPass();


    
    protected float dot(float[] a, float[] b){
        float sum = 0;

        for(int i = 0; i < a.length; i++){
            sum += a[i] * b[i];
        }

        return sum;
    }

    protected void populateParams(float min, float max){
        for(int i = 0; i < this.parameters.size(); i++){
            float[][] paramMatrix = this.parameters.get(i);

            Utility.populateRandom(paramMatrix, min, max);
        }
    }
    

    public void applyGradient(float scalar){

        for(int i = 0; i < this.gradient.size(); i++){
            float[][] gradientMatrix = this.gradient.get(i);
            float[][] paramMatrix = this.parameters.get(i);

            for(int r = 0; r < paramMatrix.length; r++){
                for(int c = 0; c < paramMatrix[r].length; c++){
                    paramMatrix[r][c] += scalar * gradientMatrix[r][c];
                }
            }
        }

        
    }
}