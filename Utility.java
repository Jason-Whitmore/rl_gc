

public class Utility {

    protected static float getRandom(float low, float high){
        float delta = high - low;

        return low + ((float) Math.random() * delta);
    }

    protected static void populateRandom(float[] array, float low, float high){
        
        for(int i = 0; i < array.length; i++){
            array[i] = Utility.getRandom(low, high);
        }
    }

    protected static void populateRandom(float[][] array, float low, float high){
        for(int i = 0; i < array.length; i++){
            Utility.populateRandom(array[i], low, high);
        }
    }

    public static float tanh(float x){
        return (float)Math.tanh((double)x);
    }

    public static float tanhPrime(float x){
        return (1f - (Utility.tanh(x) * Utility.tanh(x)));
    }
}