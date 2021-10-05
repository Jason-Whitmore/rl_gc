

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
}